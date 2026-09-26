"""開盤前的 PTT 討論量 → 當天成交量與振幅（見 PREREGISTRATION_VOLUME.md）。

一列是交易日 T，預測時點是 T 開盤（09:00）前，特徵只用那之前看得到的資訊：
  y_lv     log(成交股數)
  y_vol    log(ln(最高 / 最低))，即振幅；最高 = 最低（整天鎖死）時是缺值
  pre_abn  T-1 13:30 到 T 09:00 的文章＋留言數，log1p 後減去前 20 個交易日的平均
基準模型：HAR（前 1／5／22 天）、前一天報酬與是否收在漲跌停、星期、台指期夜盤、台積電 ADR 與費半的隔夜變動、
鉅亨新聞量、期貨結算日與 MSCI 調整日、處置期間、除權息日。
"""
import glob
import json
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from .calendar import assign_trade_date
from .prices import _roc_date

DOW = ["dow_1", "dow_2", "dow_3", "dow_4"]   # 星期二到星期五，星期一是基準
NIGHT = ["n_range", "n_abs", "n_lvol"]
US = ["adr_abs", "adr_neg", "adr_pk", "adr_lvol", "adr_none", "sox_abs", "sox_neg", "sox_pk"]


EVENTS = ["lock_up1", "lock_dn1", "disp", "exdiv"]


def baseline_cols(y: str) -> list:
    """完整的開盤前基準。另一個目標的前一天也放進來（量大的隔天振幅通常也大，反之亦然）。

    EVENTS 是會影響成交量、但不會反映在開盤跳空上的事：前一天收在漲停／跌停（沒成交的單留到今天）、
    處置期間（分盤撮合壓低成交量）、除權息日。沒發生過的股票整欄是 0，OLS 用虛擬反矩陣，係數就是 0。
    """
    other = "y_vol" if y == "y_lv" else "y_lv"
    return ([f"{y}_l1", f"{y}_l5", f"{y}_l22", f"{other}_l1", "abs_r1", "neg_r1"] + DOW + ["log_gap"]
            + NIGHT + US + ["news_abn", "settle", "msci"] + EVENTS)


def preopen_counts(times: pd.Series, days: pd.DatetimeIndex, start=None,
                   close: str = "13:30", open_: str = "09:00") -> pd.Series:
    """每個交易日 T：[T-1 收盤, T 開盤) 之間的貼文數。

    start：資料開始的時間。窗口有一部分早於 start 的日子是缺值，不是 0。
    """
    t = pd.to_datetime(times)
    td = assign_trade_date(t, days, close)
    oh, om = map(int, open_.split(":"))
    ok = td.notna() & (t < td + pd.Timedelta(hours=oh, minutes=om))
    n = td[ok].value_counts().reindex(days, fill_value=0).astype(float)
    if start is not None:
        ch, cm = map(int, close.split(":"))
        win_start = pd.Series(days, index=days).shift(1) + pd.Timedelta(hours=ch, minutes=cm)
        n[~(win_start >= pd.Timestamp(start))] = np.nan
    n.index.name = "date"
    return n


def abnormal(n: pd.Series, window: int = 20) -> pd.Series:
    """log1p(n) 減去前 window 個交易日的平均（不含當天）；不足 window 天是缺值。"""
    x = np.log1p(n)
    return x - x.shift(1).rolling(window).mean()


def load_tx_night(data_dir) -> pd.DataFrame:
    """台指期夜盤（期交所 futDataDown）。日期 T 的夜盤是 T-1 15:00 到 T 05:00，在 T 開盤前結束。

    每天取夜盤成交量最大的月契約（不含價差單）：
      n_range  log(ln(最高 / 最低))
      n_abs    |log(夜盤收盤 / 同一契約前一個日盤的收盤)|
      n_lvol   log(夜盤成交量) 減去前 20 個夜盤的平均
    """
    fs = sorted(glob.glob(f"{data_dir}/raw/taifex_tx/futDataDown_TX_*.csv"))
    d = pd.concat([pd.read_csv(f, dtype=str, index_col=False) for f in fs])
    d.columns = [c.strip() for c in d.columns]
    d["month"] = d["到期月份(週別)"].str.strip()
    d = d[(d["契約"].str.strip() == "TX") & d["month"].str.fullmatch(r"\d{6}")].copy()
    for src, col in (("最高價", "high"), ("最低價", "low"), ("收盤價", "close"), ("成交量", "volume")):
        d[col] = pd.to_numeric(d[src], errors="coerce")
    d["date"] = pd.to_datetime(d["交易日期"])
    reg = d[d["交易時段"] == "一般"].set_index(["date", "month"])["close"]
    reg = reg[~reg.index.duplicated()]
    ngt = (d[d["交易時段"] == "盤後"].sort_values("volume", na_position="first").groupby("date").tail(1)
           .set_index("date").sort_index())
    reg_days = reg.index.get_level_values("date").unique().sort_values()
    pos = reg_days.searchsorted(ngt.index) - 1                 # 前一個日盤交易日
    prev_close = reg.reindex(list(zip(reg_days[np.maximum(pos, 0)], ngt["month"]))).to_numpy(dtype=float)
    prev_close[pos < 0] = np.nan
    out = pd.DataFrame(index=ngt.index)
    out["n_range"] = np.log(np.log(ngt["high"] / ngt["low"])).replace(-np.inf, np.nan)
    out["n_abs"] = np.abs(np.log(ngt["close"].to_numpy() / prev_close))
    lv = np.log(ngt["volume"].where(ngt["volume"] > 0))
    out["n_lvol"] = lv - lv.shift(1).rolling(20).mean()
    return out


def load_us(path) -> pd.DataFrame:
    """Yahoo Finance chart API 的日資料（scripts/fetch_us.py），索引是交易所當地的交易日。"""
    r = json.load(open(path))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    idx = (pd.to_datetime(r["timestamp"], unit="s", utc=True)
           .tz_convert(r["meta"]["exchangeTimezoneName"]).tz_localize(None).normalize())
    df = pd.DataFrame({"high": q["high"], "low": q["low"], "volume": q["volume"],
                       "adj": r["indicators"]["adjclose"][0]["adjclose"]}, index=idx)
    return df.dropna(subset=["high", "low", "adj"])


def us_overnight(us: pd.DataFrame, days: pd.DatetimeIndex, prefix: str, volume: bool = True) -> pd.DataFrame:
    """每個台股交易日 T：T-1 收盤之後、T 開盤之前收盤的美股交易日。

    美股交易日 D 在台北時間 D+1 的 04:00／05:00 收盤，所以 T 開盤前看得到的最後一個是
    不晚於 T 前一個日曆日的交易日 D_last(T)；對 T 來說新的交易日是 (D_last(T-1), D_last(T)]。
    台股連假時累加多個交易日；美股休市（沒有新交易日）時變動記 0，並用 {prefix}_none 標記。
      _abs  |累積對數報酬|              _neg   累積對數報酬取負的部分
      _pk   sqrt(Σ Parkinson 變異數)    _lvol  最後一個新交易日的 log 成交量減去前 20 天平均
    """
    lr = np.log(us["adj"]).to_numpy()
    cpk = np.concatenate([[0.0], np.cumsum(np.log(us["high"] / us["low"]).to_numpy() ** 2 / (4 * np.log(2)))])
    pos = np.searchsorted(us.index.values, (days - pd.Timedelta(days=1)).values, side="right") - 1
    prev = np.concatenate([[-1], pos[:-1]])
    valid = (pos >= 0) & (prev >= 0)
    new = valid & (pos > prev)
    p0, p1 = np.maximum(prev, 0), np.maximum(pos, 0)
    ret = np.where(new, lr[p1] - lr[p0], 0.0)
    out = pd.DataFrame(index=days)
    out[f"{prefix}_abs"] = np.abs(ret)
    out[f"{prefix}_neg"] = np.minimum(ret, 0.0)
    out[f"{prefix}_pk"] = np.sqrt(np.where(new, cpk[p1 + 1] - cpk[p0 + 1], 0.0))
    if volume:
        lv = np.log(us["volume"].where(us["volume"] > 0))
        out[f"{prefix}_lvol"] = np.where(new, (lv - lv.shift(1).rolling(20).mean()).to_numpy()[p1], 0.0)
    out[f"{prefix}_none"] = (~new).astype(float)
    out.loc[~valid, :] = np.nan
    return out


def load_news(data_dir) -> pd.DataFrame:
    """鉅亨網頭條：台北時間、標題、被標記的股票代號。"""
    rows = {}
    for f in glob.glob(f"{data_dir}/raw/cnyes/headline/*/page_*.json"):
        for a in json.load(open(f, encoding="utf-8"))["items"]["data"]:
            rows[str(a["newsId"])] = (int(a["publishAt"]), a.get("title") or "", tuple(a.get("stock") or ()))
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["ts", "title", "stock"])
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    return df


def news_mentions(news: pd.DataFrame, ticker: str, aliases, exclude=()) -> pd.Series:
    """被標記成這檔股票，或標題提到別名的新聞（比對前先刪掉會撞名的詞）。"""
    title = news["title"]
    for w in exclude:
        title = title.str.replace(w, "", regex=False)
    tagged = news["stock"].map(lambda s: ticker in s)
    return tagged | title.str.contains("|".join(map(re.escape, aliases)))


def load_disposition(data_dir) -> pd.DataFrame:
    """證交所處置股（公布處置有價證券）：代號、處置期間的起訖日。

    處置在期間開始前一個交易日收盤後公告，所以期間內每一天在開盤前都已經知道。
    """
    rows = []
    for p in glob.glob(f"{data_dir}/raw/twse_punish/punish_*.json"):
        for r in json.load(open(p, encoding="utf-8")):
            a, _, b = str(r[6]).partition("～")
            try:
                rows.append((str(r[2]).strip(), _roc_date(a), _roc_date(b)))
            except (ValueError, TypeError):
                continue
    return pd.DataFrame(rows, columns=["code", "start", "end"]).drop_duplicates()


def in_periods(days: pd.DatetimeIndex, periods: pd.DataFrame) -> pd.Series:
    """落在任何一段 [start, end]（含頭尾）裡的交易日是 1。"""
    hit = np.zeros(len(days), dtype=bool)
    for s, e in zip(periods["start"], periods["end"]):
        hit |= (days >= s) & (days <= e)
    return pd.Series(hit.astype(float), index=days)


def calendar_flags(days: pd.DatetimeIndex) -> pd.DataFrame:
    """事先就知道的日子。
      settle   台指期結算日：每月第三個星期三，遇到假日順延到下一個交易日
      msci     MSCI 季度調整生效：2、5、8、11 月的最後一個交易日
      dow_1..4 星期二到星期五
      log_gap  距上一個交易日的日曆天數，取 log（週一與連假後較大）
    """
    s = pd.Series(days, index=days)
    settle = set()
    for m in pd.period_range(days.min(), days.max(), freq="M"):
        first = m.start_time
        wed3 = first + pd.Timedelta(days=(2 - first.dayofweek) % 7 + 14)
        i = days.searchsorted(wed3)
        if i < len(days) and days[i] - wed3 < pd.Timedelta(days=7):
            settle.add(days[i])
    out = pd.DataFrame(index=days)
    out["settle"] = days.isin(list(settle)).astype(float)
    last = s.groupby(days.to_period("M")).transform("max")
    out["msci"] = ((s == last) & days.month.isin([2, 5, 8, 11])).astype(float)
    for k in range(1, 5):
        out[f"dow_{k}"] = (days.dayofweek == k).astype(float)
    out["log_gap"] = np.log((s - s.shift(1)).dt.days)
    return out


def design(px: pd.DataFrame, days: pd.DatetimeIndex, pre_abn: pd.Series, *blocks: pd.DataFrame) -> pd.DataFrame:
    """個股的目標與特徵。px：panel 裡這檔股票的列（date 索引）；blocks：夜盤、美股、新聞、日曆等（date 索引）。"""
    px = px.reindex(days)
    df = pd.DataFrame(index=days)
    df["y_lv"] = np.log(px["volume"])
    df["y_vol"] = np.log(np.log(px["high"] / px["low"]).where(px["high"] > px["low"]))
    for y in ("y_lv", "y_vol"):
        s = df[y].shift(1)
        df[f"{y}_l1"], df[f"{y}_l5"], df[f"{y}_l22"] = s, s.rolling(5).mean(), s.rolling(22).mean()
    r1 = px["ret"].shift(1)
    df["abs_r1"], df["neg_r1"] = r1.abs(), r1.clip(upper=0)
    df["lock_up1"] = px["limit_up_close"].astype(float).shift(1)
    df["lock_dn1"] = px["limit_down_close"].astype(float).shift(1)
    for b in blocks:
        df = df.join(b)
    df["pre_abn"] = pre_abn
    # 開盤跳空在 T 開盤後才知道，只用來判斷訊號是不是隔夜消息的回音，不在開盤前的模型裡
    df["gap_abs"], df["gap_neg"] = px["gap"].abs(), px["gap"].clip(upper=0)
    return df


def rolling_forecast(d: pd.DataFrame, y: str, xs, first, last,
                     min_train: int = 250, refit: int = 21) -> pd.Series:
    """擴張視窗的 OLS 預測：評估期每 refit 個交易日重估一次，只用該區塊開始前的列訓練。

    要比較的兩個模型要傳入同一個 d（已經對兩個模型的所有欄位 dropna），樣本與重估時點才會一致。
    """
    ev = d.index[(d.index >= pd.Timestamp(first)) & (d.index <= pd.Timestamp(last))]
    pred = pd.Series(np.nan, index=ev)
    for i in range(0, len(ev), refit):
        block = ev[i:i + refit]
        train = d[d.index < block[0]]
        if len(train) < min_train:
            raise ValueError(f"{block[0].date()} 之前只有 {len(train)} 列可以訓練（至少要 {min_train}）")
        m = sm.OLS(train[y], sm.add_constant(train[list(xs)], has_constant="add")).fit()
        pred[block] = m.predict(sm.add_constant(d.loc[block, list(xs)], has_constant="add"))
    return pred


def cw_loss(y: pd.Series, p0: pd.Series, p1: pd.Series) -> pd.Series:
    """Clark–West（2007）調整後的損失差，> 0 表示大模型（p1）比較好。

    巢狀模型下，多出來的變數就算沒用，大模型也會因為估計誤差讓 MSE 變大；
    加回 (p0 - p1)² 修正這個偏誤，否則直接比 MSE 會偏向小模型。
    """
    return (y - p0) ** 2 - ((y - p1) ** 2 - (p0 - p1) ** 2)


def mean_test(f: pd.Series, maxlags: int = 5) -> dict:
    """平均 > 0 的單尾檢定，Newey–West 標準誤。"""
    f = f.dropna()
    m = sm.OLS(f.to_numpy(), np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    t = float(m.tvalues[0])
    return {"mean": float(f.mean()), "t": t, "p_one_sided": float(norm.sf(t)), "n": len(f)}


def expected_power(f: pd.Series, n: int, alpha: float = 0.05, maxlags: int = 5) -> dict:
    """把 f 的平均當成真的效果，換算 n 天的樣本外檢定預期的 t 與檢定力（單尾）。"""
    r = mean_test(f, maxlags)
    et = r["t"] * np.sqrt(n / r["n"])
    return {"expected_t": et, "power": float(norm.cdf(et - norm.ppf(1 - alpha)))}

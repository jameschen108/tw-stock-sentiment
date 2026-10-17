"""研究五：用槓桿 ETF 打贏大盤的總報酬。候選策略 C0–C3 與評估指標。

規則在看開發期（2003–2023）的結果之前寫死；scripts/21_leverage.py dev 會把這個檔的 sha256 寫進輸出。
兩個資產：
- x1：1 倍。2015-01-06 起是 0050；之前用 TAIEX 報酬指數扣 FEE_1X 合成（0050 偏重台積電，
  2015–2026 每年比 TAIEX TR 多約 2%，所以不拿 0050 校準，免得把台積電的超額報酬帶回 2015 年以前）。
- x2：2 倍。2015-01-06 起是 00631L；之前用台指期合成（etf.synthetic_2x），年成本 COST_2X。
權重都在 t 收盤用 t 收盤以前的資料決定，以 t 收盤價成交，t+1 生效。訊號用 TAIEX 報酬指數；
2003 年以前沒有報酬指數，暖身期間改用價格指數（只影響 2003 年初的波動與均線）。
"""
import numpy as np
import pandas as pd

from . import etf

SWITCH = pd.Timestamp("2015-01-06")   # 第一個有 0050、00631L 報酬的交易日
COST_2X = 0.0255                       # 批次 2：00631L 對期貨合成的年成本，2015–2023 校準
FEE_1X = 0.004                         # 0050 的總費用率約 0.4%/年
DEV_VOL = 0.1836                       # 開發期 2003–2023 TAIEX TR 的年化波動（日 std × √252）
TARGET_VOL = 1.5 * DEV_VOL
VOL_WINDOW, MA_WINDOW = 60, 200
BAND_C2 = 0.10
DAYS_PER_YEAR = 246                    # 台股 2003–2023 每年平均交易日；滾動視窗用


def load_inputs(cfg, panel: pd.DataFrame) -> dict:
    """讀長歷史與 ETF，回傳 bench（TR 對數報酬）、price（價格指數對數報酬）、signal、syn2x、etfs。"""
    long = cfg["work_dir"] / "long"
    tr = np.log(etf.load_twse_index(long / "tr", "MFI94U")).diff().dropna()
    price = np.log(etf.load_twse_index(long / "price", "MI_5MINS_HIST")).diff().dropna()
    rate = etf.load_cbc_rate(long / "cbc" / "EG28D01.csv")
    syn2x = etf.synthetic_2x(etf.tx_returns(etf.load_tx(long / "tx")), rate, cost=COST_2X)
    etfs = etf.etf_returns(panel, ["0050", "00631L"], etf.load_splits())
    signal = tr.combine_first(price.loc[: tr.index[0] - pd.Timedelta(days=1)])
    return {"bench": tr, "price": price, "signal": signal, "syn2x": syn2x, "etfs": etfs}


def assets(inp: dict, start, end, bench: str = "bench") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(x1, x2 的對數報酬, 能不能交易)。bench="price" 只用在 1999–2002 的描述（1x 低估股息）。"""
    b = inp[bench].loc[start:end]
    days = b.index
    days_gap = pd.Series(days, index=days).diff().dt.days.fillna(1)
    x1 = np.log1p(np.expm1(b) - FEE_1X * days_gap / 365)
    x2 = inp["syn2x"].reindex(days)
    real = days >= SWITCH
    e = inp["etfs"].reindex(days)
    x1[real], x2[real] = e.loc[real, "0050"], e.loc[real, "00631L"]
    ret = pd.DataFrame({"x1": x1, "x2": x2})
    tradable = pd.DataFrame({"x1": True, "x2": True}, index=days)
    tradable.loc[real] = e.loc[real].notna().to_numpy()
    return ret, tradable


def _first_row_filled(target: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """第一天一定要建倉：用當天算得出的目標權重。"""
    target = target.copy()
    target.iloc[0] = daily.iloc[0]
    return target


def c0(days) -> pd.DataFrame:
    return pd.DataFrame({"x1": 1.0, "x2": 0.0}, index=days)


def c1(days) -> pd.DataFrame:
    """50% x2 + 50% x1，每月第一個交易日調回。"""
    daily = pd.DataFrame({"x1": 0.5, "x2": 0.5}, index=days)
    month = pd.Series(days.to_period("M"), index=days)
    keep = month.ne(month.shift(1))
    return _first_row_filled(daily.where(keep), daily)


def c2_leverage(signal: pd.Series) -> pd.Series:
    """L = clip(目標波動 / 過去 60 天（含當天）的實現波動, 1, 2)；資料不夠 60 天時 L = 1。"""
    vol = signal.rolling(VOL_WINDOW).std() * np.sqrt(252)
    return (TARGET_VOL / vol).clip(1.0, 2.0).fillna(1.0)


def c2(signal: pd.Series, days) -> pd.DataFrame:
    """波動目標：x2 = L − 1、x1 = 2 − L（曝險 = L）。每週最後一個交易日檢查，偏離超過 BAND_C2 才調。"""
    L = c2_leverage(signal).reindex(days)
    daily = pd.DataFrame({"x1": 2 - L, "x2": L - 1}, index=days)
    week = pd.Series(days.to_period("W"), index=days)
    keep = week.ne(week.shift(-1)) & week.shift(-1).notna()   # 資料最後一天不知道是不是週末，不算
    return _first_row_filled(daily.where(keep), daily)


def c3(signal: pd.Series, days) -> pd.DataFrame:
    """趨勢：報酬指數在 200 日均線（含當天）之上持有 x2，否則持有 x1；均線算不出來時持有 x1。每天檢查。"""
    level = signal.cumsum()                 # 對數水準；比大小和用指數水準一樣
    ma = np.log(np.exp(level).rolling(MA_WINDOW).mean())
    up = (level > ma).reindex(days).fillna(False).astype(bool)
    return pd.DataFrame({"x1": (~up).astype(float), "x2": up.astype(float)}, index=days)


def strategies(signal: pd.Series, days) -> dict[str, pd.DataFrame]:
    return {"C0": c0(days), "C1": c1(days), "C2": c2(signal, days), "C3": c3(signal, days)}


def bands() -> dict[str, float]:
    return {"C0": 0.0, "C1": 0.0, "C2": BAND_C2, "C3": 0.0}


def evaluate(net: pd.Series, bench: pd.Series, nw_lags: int = 20) -> dict:
    """net：每日簡單淨報酬；bench：TR 對數報酬。年化一律用日曆天。"""
    from .event_study import newey_west_mean
    lr = np.log1p(net)
    b = bench.reindex(lr.index)
    ex = lr - b
    years = (lr.index[-1] - lr.index[0]).days / 365.25
    eq = np.exp(lr.cumsum())
    roll = ex.rolling(3 * DAYS_PER_YEAR).sum().dropna()
    nw = newey_west_mean(ex, nw_lags)
    return {
        "cagr": np.expm1(lr.sum() / years), "bench_cagr": np.expm1(b.sum() / years),
        "excess_cagr": np.expm1(lr.sum() / years) - np.expm1(b.sum() / years),
        "ann_vol": lr.std() * np.sqrt(252), "max_dd": (eq / eq.cummax() - 1).min(),
        "worst_12m": np.expm1(lr.rolling(DAYS_PER_YEAR).sum().min()),
        "win_3y": (roll > 0).mean(), "excess_log_ann": ex.mean() * 252, "nw_t": nw["t"],
    }


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm 逐步校正：p 由小到大，第 k 個（從 0 起）和 alpha / (m − k) 比，第一個沒過之後全部不過。"""
    out, m, ok = {}, len(pvals), True
    for k, (name, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        ok = ok and p < alpha / (m - k)
        out[name] = ok
    return out


def bootstrap_win(excess: pd.Series, horizon: int = 3 * DAYS_PER_YEAR, block: int = 60,
                  reps: int = 5000, seed: int = 0) -> float:
    """stationary block bootstrap（平均區塊 60 天）：horizon 天內累積超額對數報酬 > 0 的機率。

    把策略的每日超額報酬當成固定的序列重抽，近似「隨便挑一段 3 年」的勝率；策略本身依賴路徑，所以只是近似。
    """
    x = excess.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    k = np.arange(horizon)
    new = rng.random((reps, horizon)) < 1 / block
    new[:, 0] = True
    k_start = np.maximum.accumulate(np.where(new, k, 0), axis=1)     # 每一天所屬區塊的起點
    starts = np.take_along_axis(rng.integers(n, size=(reps, horizon)), k_start, axis=1)
    idx = (starts + k - k_start) % n
    return float((x[idx].sum(axis=1) > 0).mean())

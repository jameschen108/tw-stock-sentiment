"""ETF 的日報酬，給研究五（槓桿 ETF）用。

面板（panel.py）把 |報酬| > 10.5% 設成缺值，但 2 倍 ETF 沒有 10% 的漲跌幅限制，大跌、大漲日會被刪掉
（00631L 有 17 天）。這裡從面板的收盤與基準價（已含除息參考價）重算，不過濾。
分割恢復買賣那天，基準價（停牌前最後一個收盤）除以 docs/leverage/splits.csv 的分割比例。
停止買賣的日子沒有資料列，所以恢復買賣那天的報酬涵蓋整段停牌期間。

2015 年以前沒有 00631L，用台指期合成（synthetic_2x）：2 倍 ETF 持有的是台指期，
大跌日期貨（13:45 結算）常比現貨（13:30 收盤）跌得更多，用現貨合成會低估崩盤損失。
長歷史由 scripts/fetch_long_history.py 下載到 data/long/。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = Path(__file__).resolve().parents[2] / "docs" / "leverage" / "splits.csv"


def load_splits(path=SPLITS) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"code": str}, parse_dates=["date", "last_trade"])


def etf_returns(panel: pd.DataFrame, codes, splits: pd.DataFrame) -> pd.DataFrame:
    """回傳 日期 × 代號 的對數報酬；每檔第一天、沒有成交的日子是缺值。"""
    out = {}
    for code in codes:
        s = panel.loc[panel["code"] == code, ["date", "close", "base"]].set_index("date").sort_index()
        base = s["base"].copy()
        for _, sp in splits[splits["code"] == code].iterrows():
            if sp["date"] in base.index:        # 資料只取一段期間時，分割日可能不在裡面
                base.loc[sp["date"]] /= sp["ratio"]
        out[code] = np.log(s["close"] / base)
    return pd.DataFrame(out).sort_index()


def load_tx(tx_dir) -> pd.DataFrame:
    """FinMind 台指期日資料 -> 一般交易時段、月契約（不含價差、週契約）的 date, contract, settle, close。"""
    rows = []
    for p in sorted(Path(tx_dir).glob("TaiwanFuturesDaily_TX_[0-9][0-9][0-9][0-9].json")):
        rows += json.loads(p.read_text())["data"]
    d = pd.DataFrame(rows)
    d = d[(d["trading_session"] == "position") & d["contract_date"].str.fullmatch(r"\d{6}")]
    d = d.rename(columns={"contract_date": "contract", "settlement_price": "settle"})
    d["date"] = pd.to_datetime(d["date"])
    return d[["date", "contract", "settle", "close"]].sort_values(["date", "contract"]).reset_index(drop=True)


def tx_returns(tx: pd.DataFrame) -> pd.Series:
    """連續近月台指期的每日對數報酬（結算價對結算價）。

    第 i 天的報酬用「到期日晚於第 i 天」的最近月契約：到期前一天收盤就換到下一個月，
    所以不會用到到期日當天的最後結算價（FinMind 的結算價在到期日是 0）。
    到期日 = 契約在資料裡的最後一天；最後一天就是資料最後一天的契約當作還沒到期。
    """
    s = tx.set_index(["date", "contract"])["settle"].unstack()
    exp = s.apply(lambda c: c.last_valid_index())
    exp[exp == s.index[-1]] = pd.Timestamp("2200-01-01")
    out = {}
    for prev, d in zip(s.index[:-1], s.index[1:]):
        held = exp[exp > d].sort_values().index[0]
        a, b = s.at[prev, held], s.at[d, held]
        if not (a > 0 and b > 0):
            raise ValueError(f"{d.date()} 換倉的契約 {held} 前一天或當天沒有結算價")
        out[d] = np.log(b / a)
    return pd.Series(out, name="tx")


def compare_tx_official(tx: pd.DataFrame, official_dir) -> pd.DataFrame:
    """和期交所官方 futDataDown（共用資料夾 raw/taifex_tx/）比對結算價：回傳兩邊都有的列與差距。"""
    frames = []
    for p in sorted(Path(official_dir).glob("futDataDown_TX_[0-9]*.csv")):
        o = pd.read_csv(p, encoding="utf-8-sig", index_col=False, dtype=str)
        o = o[(o["交易時段"].str.strip() == "一般") & o["到期月份(週別)"].str.strip().str.fullmatch(r"\d{6}")]
        frames.append(pd.DataFrame({"date": pd.to_datetime(o["交易日期"].str.strip()),
                                    "contract": o["到期月份(週別)"].str.strip(),
                                    "official": pd.to_numeric(o["結算價"].str.strip(), errors="coerce")}))
    m = tx.merge(pd.concat(frames), on=["date", "contract"])
    m["diff"] = m["settle"] - m["official"]
    return m


def load_twse_index(index_dir, prefix: str) -> pd.Series:
    """證交所月檔 -> 每日收盤指數。prefix：MFI94U（報酬指數）或 MI_5MINS_HIST（價格指數）。"""
    out = {}
    for p in sorted(Path(index_dir).glob(f"{prefix}_[0-9]*.json")):
        js = json.loads(p.read_text())
        for r in js.get("data") or []:
            y, m, d = str(r[0]).strip().split("/")
            out[pd.Timestamp(int(y) + 1911, int(m), int(d))] = float(str(r[-1]).replace(",", ""))
    return pd.Series(out, name=prefix).sort_index()


def load_cbc_rate(path) -> pd.Series:
    """央行重貼現率（%），只有調整日；之後的日子用 ffill。"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    return pd.Series(df.iloc[:, 1].astype(float).to_numpy(),
                     index=pd.to_datetime(df.iloc[:, 0], format="%Y%m%d"), name="rate").sort_index()


def synthetic_2x(fut_ret: pd.Series, rate: pd.Series, cost: float = 0.0, mult: float = 2.0) -> pd.Series:
    """用期貨合成每日重設的槓桿 ETF：保證金現金賺利率，加 mult 倍的期貨報酬，扣年成本 cost。

    簡單報酬 = 前一天利率 × 天數/365 + mult × 期貨簡單報酬 − cost × 天數/365，回傳對數報酬。
    cost 吸收管理費、交易成本，以及 ETF 現金實際利率和重貼現率的差距。
    """
    days = pd.Series(fut_ret.index, index=fut_ret.index).diff().dt.days
    r = rate.reindex(rate.index.union(fut_ret.index)).ffill().shift(1).reindex(fut_ret.index) / 100
    simple = r * days / 365 + mult * np.expm1(fut_ret) - cost * days / 365
    return np.log1p(simple).iloc[1:]


def check_vs_index(ret: pd.Series, index_ret: pd.Series, mult: float, tol: float = 0.03) -> pd.DataFrame:
    """列出 |ETF 報酬 − 倍數 × 大盤報酬| > tol 的日子。

    ETF 停牌時沒有資料列，大盤報酬就從 ETF 上一筆資料的隔天累加到當天，和 ETF 報酬涵蓋同一段期間。
    """
    r = ret.dropna()
    cum = index_ret.fillna(0.0).cumsum()
    prev = pd.Series(r.index, index=r.index).shift(1)
    idx = cum.reindex(r.index).to_numpy() - cum.reindex(prev).to_numpy()
    df = pd.DataFrame({"etf": r, "index": mult * idx}, index=r.index)
    df["diff"] = df["etf"] - df["index"]
    df["days"] = [len(cum.loc[p:d]) - 1 if pd.notna(p) else np.nan for p, d in zip(prev, r.index)]
    return df[df["diff"].abs() > tol]

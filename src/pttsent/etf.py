"""ETF 的日報酬，給研究五（槓桿 ETF）用。

面板（panel.py）把 |報酬| > 10.5% 設成缺值，但 2 倍 ETF 沒有 10% 的漲跌幅限制，大跌、大漲日會被刪掉
（00631L 有 17 天）。這裡從面板的收盤與基準價（已含除息參考價）重算，不過濾。
分割恢復買賣那天，基準價（停牌前最後一個收盤）除以 docs/leverage/splits.csv 的分割比例。
停止買賣的日子沒有資料列，所以恢復買賣那天的報酬涵蓋整段停牌期間。
"""
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

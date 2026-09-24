"""股價、除權息還原與大盤指數。

報酬一律用對數報酬。FinMind 的日價未還原，除權息日的報酬改用
「收盤 / 除權息參考價」計算，否則配息日會出現假的下跌。
"""
import glob
import json
import re
import warnings

import numpy as np
import pandas as pd


def _num(s):
    return float(str(s).replace(",", ""))


def _roc_date(s):
    """'103/01/02' 或 '107年07月02日' -> Timestamp"""
    y, m, d = (int(x) for x in re.findall(r"\d+", s)[:3])
    return pd.Timestamp(y + 1911, m, d)


def load_taiex(data_dir) -> pd.DataFrame:
    """TWSE 發行量加權股價指數，日頻開高低收。"""
    rows = {}
    for p in sorted(glob.glob(f"{data_dir}/raw/twse_taiex/MI_5MINS_HIST_*.json")):
        for r in json.load(open(p)).get("data") or []:
            rows[_roc_date(r[0])] = [_num(x) for x in r[1:5]]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["open", "high", "low", "close"])
    df = df.sort_index()
    df.index.name = "date"
    return df


def trading_days(data_dir) -> pd.DatetimeIndex:
    return load_taiex(data_dir).index


def load_exrights(data_dir, ticker) -> pd.Series:
    """除權息日 -> 除權息參考價。欄位：[日期, 代號, 名稱, 除權息前收盤價, 除權息參考價, ...]"""
    ref = {}
    for p in sorted(glob.glob(f"{data_dir}/raw/twse_exrights/twt49u_*.json")):
        for r in json.load(open(p)):
            if str(r[1]).strip() == ticker:
                ref[_roc_date(r[0])] = _num(r[4])
    return pd.Series(ref, dtype=float).sort_index()


def load_stock(data_dir, ticker) -> pd.DataFrame:
    """個股日價（FinMind）加上還原後的對數報酬。

    欄位：open high low close volume value ret oc gap
      ret : 收盤對前一日收盤（除權息日改對參考價）
      oc  : 當日開盤到收盤
      gap : 開盤對前一日收盤（同樣處理除權息）
    """
    raw = json.load(open(f"{data_dir}/raw/finmind/price/{ticker}.json"))
    df = pd.DataFrame(raw["data"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"max": "high", "min": "low",
                            "Trading_Volume": "volume", "Trading_money": "value"})
    df = df[(df["close"] > 0) & (df["volume"] > 0)]
    df = df.set_index("date").sort_index()[["open", "high", "low", "close", "volume", "value"]]

    base = df["close"].shift(1)
    ex = load_exrights(data_dir, ticker).reindex(df.index).dropna()
    base.loc[ex.index] = ex
    df["ret"] = np.log(df["close"] / base)
    df["oc"] = np.log(df["close"] / df["open"])
    df["gap"] = np.log(df["open"] / base)

    jumps = df.index[np.abs(np.expm1(df["ret"])) > 0.105]
    if len(jumps):
        warnings.warn(f"{ticker} 有 {len(jumps)} 天報酬超過漲跌幅限制（可能是減資或資料錯誤）："
                      f"{[d.date().isoformat() for d in jumps[:5]]}")
    return df

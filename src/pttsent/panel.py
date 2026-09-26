"""全市場日價面板：上市（證交所 MI_INDEX）＋上櫃（櫃買 dailyQuotes），由 scripts/fetch_prices.py 下載。

報酬一律是對數報酬，除權息、減資都已還原：
- 上市：基準是前一個交易日收盤；除權息日改用 TWT49U 的參考價，減資恢復買賣日改用 TWTAUU 的參考價
  （兩者都在共用資料夾 raw/ 底下）。
- 上櫃：基準直接用前一筆資料的「次日參考價」，除權息、減資都已經算進去。
仍然超過漲跌幅限制的報酬（多半是新上市前幾天或沒抓到的調整）設成缺值。

大盤：上市用「發行量加權股價報酬指數」（含息，和還原後的個股報酬一致）；
上櫃用當天所有上櫃普通股依前一日市值（發行股數 × 基準價）加權的報酬。
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .prices import _roc_date

COMMON = re.compile(r"[1-9]\d{3}")          # 普通股：四位數、不是 0 開頭（0 開頭是 ETF）
KEEP = re.compile(r"\d{4}|00\d{2,4}[A-Z]?")  # 普通股與 ETF；權證、債券等不收


def _f(x):
    """'1,234.5' -> 1234.5；'--'、'---'、空字串 -> NaN"""
    s = str(x).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _table(js, key_field):
    for t in js.get("tables") or []:
        if key_field in (t.get("fields") or []):
            yield t


def parse_twse(path: Path):
    js = json.loads(path.read_text(encoding="utf-8"))
    date = pd.Timestamp(path.stem)
    rows = []
    for t in _table(js, "證券代號"):
        f = {k: i for i, k in enumerate(t["fields"])}
        for r in t["data"]:
            code = str(r[f["證券代號"]]).strip()
            if not KEEP.fullmatch(code):
                continue
            rows.append((date, code, str(r[f["證券名稱"]]).strip(), "TWSE", _f(r[f["開盤價"]]), _f(r[f["最高價"]]),
                         _f(r[f["最低價"]]), _f(r[f["收盤價"]]), _f(r[f["成交股數"]]), _f(r[f["成交金額"]]),
                         np.nan, np.nan, np.nan))
    idx = {}
    for t in js.get("tables") or []:
        fields = t.get("fields") or []
        if fields[:1] in (["指數"], ["報酬指數"]):
            for r in t.get("data") or []:
                if str(r[0]).strip() in ("發行量加權股價指數", "發行量加權股價報酬指數"):
                    idx[str(r[0]).strip()] = _f(r[1])
    return rows, (date, idx.get("發行量加權股價指數", np.nan), idx.get("發行量加權股價報酬指數", np.nan))


def parse_tpex(path: Path):
    js = json.loads(path.read_text(encoding="utf-8"))
    date = pd.Timestamp(path.stem)
    rows = []
    for t in _table(js, "代號"):
        f = {k.replace(" ", ""): i for i, k in enumerate(t["fields"])}
        for r in t["data"]:
            code = str(r[f["代號"]]).strip()
            if not KEEP.fullmatch(code):
                continue
            rows.append((date, code, str(r[f["名稱"]]).strip(), "TPEx", _f(r[f["開盤"]]), _f(r[f["最高"]]),
                         _f(r[f["最低"]]), _f(r[f["收盤"]]), _f(r[f["成交股數"]]), _f(r[f["成交金額(元)"]]),
                         _f(r[f["發行股數"]]), _f(r[f["次日參考價"]]), _f(r[f["次日漲停價"]])))
    return rows


COLS = ["date", "code", "name", "market", "open", "high", "low", "close", "volume", "value",
        "shares", "ref_next", "limit_up_next"]


def load_raw(work_dir: Path):
    """讀所有已下載的日檔 -> (面板 DataFrame, 上市指數 DataFrame)。"""
    rows, idx = [], []
    for p in sorted((work_dir / "prices" / "twse").glob("*.json")):
        r, i = parse_twse(p)
        rows += r
        idx.append(i)
    for p in sorted((work_dir / "prices" / "tpex").glob("*.json")):
        rows += parse_tpex(p)
    df = pd.DataFrame(rows, columns=COLS).drop_duplicates(["date", "code", "market"])
    twse_idx = pd.DataFrame(idx, columns=["date", "taiex", "taiex_tr"]).set_index("date").sort_index()
    return df.sort_values(["code", "date"]).reset_index(drop=True), twse_idx


def _twse_refs(data_dir) -> pd.Series:
    """(日期, 代號) -> 當天的參考價：除權息（TWT49U）與減資恢復買賣（TWTAUU）。"""
    ref = {}
    for p in sorted(glob.glob(f"{data_dir}/raw/twse_exrights/twt49u_*.json")):
        for r in json.load(open(p, encoding="utf-8")):
            ref[(_roc_date(r[0]), str(r[1]).strip())] = _f(r[4])
    for p in sorted(glob.glob(f"{data_dir}/raw/twse_reduction/twtauu_*.json")):
        js = json.load(open(p, encoding="utf-8"))
        for r in (js if isinstance(js, list) else js.get("data") or []):
            ref[(_roc_date(r[0]), str(r[1]).strip())] = _f(r[4])
    idx = pd.MultiIndex.from_tuples(list(ref), names=["date", "code"])
    return pd.Series(list(ref.values()), index=idx, dtype=float).dropna()   # 參考價是「-」的沒有用


def build(work_dir: Path, data_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳（面板, 大盤）。面板每列一檔股票一個交易日，只保留有成交的日子。

    面板欄位另外有：base（今天報酬的基準價）、ret（收盤對基準）、oc（開盤到收盤）、gap（開盤對基準）、
    limit_up_open / limit_up_close（開盤／收盤就是漲停，買不到）、limit_down_open / limit_down_close、
    common（普通股）、mkt_ret（所屬市場的大盤報酬）。
    """
    df, twse_idx = load_raw(work_dir)
    df = df[(df["close"] > 0) & (df["volume"] > 0)].copy()
    g = df.groupby(["code", "market"], sort=False)
    prev_close = g["close"].shift(1)
    prev_ref = g["ref_next"].shift(1)
    prev_up = g["limit_up_next"].shift(1)

    refs = _twse_refs(data_dir)
    key = pd.MultiIndex.from_arrays([df["date"], df["code"]])
    twse_ref = pd.Series(refs.reindex(key).to_numpy(), index=df.index)
    is_twse = df["market"] == "TWSE"
    df["base"] = np.where(is_twse, twse_ref.fillna(prev_close), prev_ref.fillna(prev_close))

    df["ret"] = np.log(df["close"] / df["base"])
    df["oc"] = np.log(df["close"] / df["open"])
    df["gap"] = np.log(df["open"] / df["base"])
    bad = np.abs(np.expm1(df["ret"])) > 0.105
    df.loc[bad, ["ret", "gap"]] = np.nan
    up = np.where(is_twse, df["base"] * 1.095, prev_up * 0.999)
    df["limit_up_open"] = df["open"] >= up
    df["limit_down_open"] = df["open"] <= df["base"] * 0.905
    df["limit_up_close"] = df["close"] >= up        # 收盤漲停：以收盤價進場通常買不到
    df["limit_down_close"] = df["close"] <= df["base"] * 0.905
    df["common"] = df["code"].str.fullmatch(COMMON.pattern) & ~df["name"].str.endswith("-DR")   # 存託憑證不算

    # 上櫃大盤：普通股依前一日市值加權
    otc = df[(df["market"] == "TPEx") & df["common"] & df["ret"].notna()].copy()
    otc["w"] = otc.groupby("code")["shares"].shift(1) * otc["base"]
    otc = otc.dropna(subset=["w"])
    otc_ret = (otc["w"] * np.expm1(otc["ret"])).groupby(otc["date"]).sum() / otc.groupby("date")["w"].sum()
    mkt = pd.DataFrame({"TWSE": np.log(twse_idx["taiex_tr"]).diff(), "TPEx": np.log1p(otc_ret)})
    mkt["taiex"] = np.log(twse_idx["taiex"]).diff()
    mkt.index.name = "date"

    m = mkt[["TWSE", "TPEx"]].stack().rename("mkt_ret")
    m.index.names = ["date", "market"]
    df = df.join(m, on=["date", "market"])
    return df.reset_index(drop=True), mkt

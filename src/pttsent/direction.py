"""開盤前的情緒能不能預告漲跌方向（docs/large_moves_direction.md）。

文字比對沿用 large_moves.mentions；情緒分數沿用研究一的 BERT 分類器（sentiment/bert.py）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import large_moves as lm

MIN_ACCOUNTS = 3     # 有分數的帳號少於這個數，sent 是缺值
D2_ABN_MIN = 1.0     # D2：開盤前（到 08:30）的 PTT 異常量至少這麼高


def preopen_texts(ptt_dir: Path, years, rules: pd.DataFrame, days: pd.DatetimeIndex, max_lag_days: int,
                  start: str, end: str) -> pd.DataFrame:
    """每檔在「t−1 收盤到 t 的 08:30」之間的 PTT 文字（比照研究一的 select_texts）。

    回傳欄位：code、pos（t 在交易日曆上的位置）、time、account、kind、text、title。
    文章的 text 是標題加內文；留言的 text 是留言本身，title 是所在文章的標題（給 BERT 當上下文）。
    """
    out = []
    for y in years:
        a = pq.read_table(ptt_dir / f"articles_{y}.parquet",
                          columns=["article_id", "time", "author", "title", "content"]).to_pandas()
        c = pq.read_table(ptt_dir / f"comments_{y}.parquet",
                          columns=["article_id", "time", "user", "content", "lag_sec"]).to_pandas()
        a = a[(a["time"] >= start) & (a["time"] < end)]
        c = c[(c["time"] >= start) & (c["time"] < end) & (c["lag_sec"] <= max_lag_days * 86400)]
        # 只留開盤前（到 08:30）的文字，先篩掉大部分，比對才快
        a = a[_early(a["time"], days)]
        c = c[_early(c["time"], days)]
        all_titles = pq.read_table(ptt_dir / f"articles_{y}.parquet", columns=["article_id", "title"]).to_pandas()
        thread_title = dict(zip(all_titles["article_id"], all_titles["title"]))
        a_title = lm.norm_text(a["title"].fillna(""))
        c_content = lm.norm_text(c["content"].fillna(""))
        c_thread = lm.norm_text(c["article_id"].map(thread_title).fillna(""))
        for r in rules.itertuples():
            ta = lm.mentions(a_title, r)
            tc = lm.mentions(c_thread, r) | lm.mentions(c_content, r)
            sa, sc = a[ta], c[tc]
            out.append(pd.DataFrame({"code": r.code, "time": sa["time"], "account": sa["author"], "kind": "article",
                                     "text": sa["title"].fillna("") + "\n" + sa["content"].fillna(""), "title": None}))
            out.append(pd.DataFrame({"code": r.code, "time": sc["time"], "account": sc["user"], "kind": "comment",
                                     "text": sc["content"].fillna(""), "title": sc["article_id"].map(thread_title)}))
    d = pd.concat(out, ignore_index=True)
    d["pos"] = lm.day_parts(d["time"], days)["pos"].reindex(range(len(d))).to_numpy()
    return d.dropna(subset=["pos"]).astype({"pos": int})


def _early(times: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    """發布時間落在某個交易日 t 的「t−1 收盤到 t 的 08:30」之間。"""
    parts = lm.day_parts(times, days)
    t = pd.to_datetime(times.reset_index(drop=True)).to_numpy()[parts.index]
    cut = (days[parts["pos"]] + pd.Timedelta(hours=lm.TRIAL[0], minutes=lm.TRIAL[1])).to_numpy()
    ok = np.zeros(len(times), dtype=bool)
    ok[parts.index] = parts["pre"].to_numpy() & (t < cut)
    return ok


def daily_sent(texts: pd.DataFrame) -> pd.DataFrame:
    """每個（code, pos）：帳號內先平均、再跨帳號平均的情緒（比照研究一的 daily_sentiment），以及有分數的帳號數。"""
    s = texts.dropna(subset=["score"])
    unit = s.groupby(["code", "pos", "account"])["score"].mean()
    g = unit.groupby(level=["code", "pos"])
    out = pd.DataFrame({"sent_raw": g.mean(), "n_accounts": g.size()})
    out["sent"] = out["sent_raw"].where(out["n_accounts"] >= MIN_ACCOUNTS)
    return out.reset_index()

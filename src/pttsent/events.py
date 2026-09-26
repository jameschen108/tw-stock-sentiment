"""PTT [標的] 文 -> 事件表：作者自標多空、對到一檔台股、對齊交易日。

來源有兩個，用 article_id 合併：
- pttcc（data/interim/ptt/articles_{年}.parquet，2019–2024）：有作者、推噓數。
- pttweb（共用資料夾 pttweb/，2015/04–2025/01）：多了 2016–2018，也抓到被刪的文章（標題帶「已刪文」）；
  沒有作者，留言只有文字。

時間對齊（exit_idx 等都是交易日曆上的位置）：
- entry：發文後第一個收盤。13:30 以前發的文是當天收盤，之後（或非交易日）是下一個交易日收盤。
- 發文前最後一個收盤是 entry 的前一個交易日；「發文當下」的報酬就是 entry 那一天的報酬。
- 發文後 h 天的報酬：entry 收盤到之後第 h 個交易日收盤，全部是收盤對收盤，和大盤一致。
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import assign_trade_date
from .ptt import strip_quotes
from .sentiment.weak_labels import label_from_content

INDEX_RE = re.compile(r"大盤|台指|加權指數|期指|小台|微台|台股指數")
CODE_RE = re.compile(r"(?<![0-9A-Za-z])(\d{4}|00\d{2,4}[A-Z]?)(?![0-9])")
TITLE_DIR_RE = re.compile(r"(?:^|[\s(（\]】\-－/])(多|空)(?:$|[\s)）】!！?？,，.。])")
YEAR_LIKE = re.compile(r"(19|20)\d\d")


def load_pttweb_targets(pttweb_dir: Path) -> pd.DataFrame:
    """掃 pttweb 所有檔案，只留 [標的] 原文（不含 Re:）。約 25 萬個檔，要一兩分鐘。"""
    rows = []
    for b in sorted(os.listdir(pttweb_dir)):
        if not b.startswith("batch"):
            continue
        for f in os.listdir(pttweb_dir / b):
            if not f.endswith(".json"):
                continue
            d = json.load(open(pttweb_dir / b / f, encoding="utf-8"))
            title = d.get("title") or ""
            if not title.startswith("[標的]"):
                continue
            ts = d.get("timestamp") or int(d["article_id"].split(".")[1])   # 文章 ID 本身就是發文的 unix 時間
            rows.append({"article_id": d["article_id"], "timestamp": ts, "title": title,
                         "body": d.get("body") or "", "n_comments_web": len(d.get("pushes") or []),
                         "pushes": d.get("pushes") or []})
    df = pd.DataFrame(rows)
    df["time"] = (pd.to_datetime(df["timestamp"], unit="s", utc=True)
                  .dt.tz_convert("Asia/Taipei").dt.tz_localize(None))
    return df.drop(columns="timestamp")


def load_pttcc_targets(ptt_dir: Path, years) -> pd.DataFrame:
    parts = []
    for y in years:
        a = pd.read_parquet(ptt_dir / f"articles_{y}.parquet")
        parts.append(a[(a["category"] == "標的") & ~a["is_reply"]])
    a = pd.concat(parts, ignore_index=True)
    return a[["article_id", "time", "author", "title", "content", "n_push", "n_boo"]]


def combine(cc: pd.DataFrame, web: pd.DataFrame) -> pd.DataFrame:
    """合併兩個來源。兩邊都有的文章：時間、作者、推噓用 pttcc，內文用 pttcc（已去掉引述）。"""
    web = web.assign(deleted=web["title"].str.contains("已刪文"),
                     title=web["title"].str.replace(r"\s*已刪文\s*$", "", regex=True),
                     content=web["body"].map(strip_quotes))
    m = cc.merge(web[["article_id", "time", "title", "content", "deleted", "n_comments_web"]],
                 on="article_id", how="outer", suffixes=("", "_web"), indicator=True)
    for c in ("time", "title", "content"):
        m[c] = m[c].fillna(m[f"{c}_web"])
    m["in_pttcc"] = m["_merge"] != "right_only"
    m["in_pttweb"] = m["_merge"] != "left_only"
    m["deleted"] = m["deleted"].astype("boolean").fillna(False).astype(bool)
    keep = ["article_id", "time", "author", "title", "content", "n_push", "n_boo", "n_comments_web",
            "in_pttcc", "in_pttweb", "deleted"]
    return m[keep].sort_values("time").reset_index(drop=True)


def title_direction(title: str):
    found = {m.group(1) for m in TITLE_DIR_RE.finditer(title.replace("[標的]", " "))}
    if found == {"多"}:
        return "bullish"
    if found == {"空"}:
        return "bearish"
    return None


def listed_names(panel: pd.DataFrame) -> pd.DataFrame:
    """每個（代號, 名稱）出現的期間：名稱會改（例如台新金 -> 台新新光金），要依發文日期比對。"""
    g = panel.groupby(["code", "name"])["date"]
    return pd.DataFrame({"first": g.min(), "last": g.max()}).reset_index()


class Resolver:
    """標題 -> 一檔台股代號。規則見 resolve 的說明。"""

    def __init__(self, names: pd.DataFrame, market_of: dict):
        self.names = names
        self.market_of = market_of

    def active(self, date):
        pad = pd.Timedelta(days=10)
        a = self.names[(self.names["first"] <= date + pad) & (self.names["last"] >= date - pad)]
        return dict(zip(a["code"], a["name"])), a

    def resolve(self, title: str, date):
        """回傳（代號 or None, 方法或排除原因）。

        - 標題裡的四位數或 ETF 代號，當天有在交易才算；像年份的數字（19xx、20xx）要標題裡也有它的名稱才算。
        - 名稱比對：長名稱優先，比對到的部分先挖掉（「長榮航」不會再算成「長榮」）。
        - 代號和名稱指向同一檔 -> 成功；指向兩檔以上 -> 排除（multi）。
        """
        t = title.replace("[標的]", " ")
        code_name, act = self.active(date)
        codes = []
        for m in CODE_RE.finditer(t):
            c = m.group(1)
            if c not in code_name:
                continue
            if YEAR_LIKE.fullmatch(c) and code_name[c] not in t:
                continue
            codes.append(c)
        rest = CODE_RE.sub(" ", t)
        by_name = []
        pairs = [(n, c) for n, c in zip(act["name"], act["code"])]
        pairs += [(n[:-3], c) for n, c in pairs if n.endswith("-KY")]   # 標題常省略 -KY
        for name, code in sorted(pairs, key=lambda x: -len(x[0])):
            if len(name) >= 2 and name in rest:
                by_name.append(code)
                rest = rest.replace(name, " ")
        cand = set(codes) | set(by_name)
        if not cand:
            return None, "index" if INDEX_RE.search(t) else "no_tw_stock"
        if len(cand) > 1:
            return None, "multi"
        c = cand.pop()
        how = "code+name" if (codes and by_name) else ("code" if codes else "name")
        return c, how


def entry_positions(times: pd.Series, days: pd.DatetimeIndex, cutoff: str = "13:30") -> np.ndarray:
    """每則發文 -> entry（發文後第一個收盤）在交易日曆上的位置；超出日曆 -> -1。
    和每日情緒用的是同一個對齊規則（calendar.assign_trade_date）。"""
    return days.get_indexer(assign_trade_date(times, days, cutoff))

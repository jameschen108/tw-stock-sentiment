"""大漲跌的原因分類（docs/large_moves.md 第三部分）：把事件前後的鉅亨新聞、重大訊息、PTT 標題交給 LLM 判斷主要原因。

API 呼叫沿用 sentiment/llm_label.py 的 client、用量與費用計算；這裡只放輸入怎麼組、提示詞與輸出格式。
"""
import glob
import html
import json
import re

import numpy as np
import pandas as pd

from .calendar import assign_trade_date

CATEGORIES = ["月營收", "財報財測法說", "公司重大事件", "產業題材", "法人籌碼", "只描述股價", "其他"]
NO_NEWS = "找不到相關新聞"          # 窗口內沒有鉅亨新聞也沒有重大訊息：不送 LLM，直接歸這類
NOT_FOUND = "其他：找不到原因"       # 報表用：LLM 選「其他」而且 cause_ids 是空的
CONTENT_CHARS = 300
CAP = {"news": 25, "mops": 15, "ptt": 15}   # 每個來源最多幾則，取離事件日收盤最近的
SOURCE_NAME = {"news": "鉅亨", "mops": "重大訊息", "ptt": "PTT"}

SYSTEM = """你是台灣股市的研究助理。給你一檔股票在某個交易日 t 前後一週的新聞、公司重大訊息與 PTT 文章標題，判斷 t 當天股價大漲或大跌的主要原因屬於哪一類。

類別：
- 月營收：月營收公布、營收創新高或衰退
- 財報財測法說：季報、年報、自結損益、EPS、財測、法說會
- 公司重大事件：訂單、併購、增資減資、庫藏股、股利、經營權、訴訟、事故、停工、新產品、處分資產
- 產業題材：政策、原物料或產品報價、概念股、同業或客戶的消息、整個產業的利多利空
- 法人籌碼：外資或投信報告與評等、目標價、法人買賣超、融資融券、注意股與處置
- 只描述股價：只有「亮燈」「強攻」「重挫」之類描述股價本身的報導，沒有說明原因
- 其他：有和這家公司相關的消息，但不屬於上面任何一類

規則：
- 只根據提供的資料判斷，不要使用你對這段期間實際行情或公司的其他知識。
- 原因要能解釋 t 當天的漲跌方向：利多對應大漲、利空對應大跌。方向不符的消息不算原因。
- 例行公告（代子公司取得理財商品、背書保證、董事會開會通知等）通常不是原因。
- 有好幾則相關消息時，選最能解釋當天漲跌的那一類；cause_ids 列出支持判斷的資料編號（最多 3 個）。
- 資料裡只有描述股價的報導、找不到任何原因時，選「只描述股價」；連這種報導都沒有、也沒有相關消息時，選「其他」並把 cause_ids 留空。
- 資料可能在 t 之後才發布（例如盤後解釋當天漲跌的報導），一樣可以當作原因的依據；發布時間由研究者另外判斷，你不用管。
- reason 用 40 字以內說明判斷理由。
- confidence：high 是資料明確指出原因；medium 是有合理的相關消息但沒有明說；low 是勉強推測。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "cause_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": ["category", "cause_ids", "confidence", "reason"],
    "additionalProperties": False,
}


def _clean_content(c: str) -> str:
    t = html.unescape(html.unescape(c or ""))
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\s+", " ", t).strip()[:CONTENT_CHARS]


def load_news_content(data_dir) -> pd.DataFrame:
    """鉅亨頭條：newsId、台北時間、標題、內文前 300 字。"""
    rows = {}
    for f in glob.glob(f"{data_dir}/raw/cnyes/headline/*/page_*.json"):
        for a in json.load(open(f, encoding="utf-8"))["items"]["data"]:
            rows[str(a["newsId"])] = (int(a["publishAt"]), a.get("title") or "", _clean_content(a.get("content")))
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["ts", "title", "content"])
    df.index.name = "news_id"
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
    return df.drop(columns="ts").reset_index()


def window_items(items: pd.DataFrame, days: pd.DatetimeIndex, d: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """每個（code, pos）事件：items 中屬於交易日位置 pos+lo .. pos+hi（含）的列。

    items 要有 code、time；交易日依 13:30 收盤切（收盤後算下一個交易日）。lo=-5、hi=1 就是 t−6 收盤到 t+1 收盤。
    回傳 items 的列加上 ev（d 的索引）。
    """
    td = assign_trade_date(items["time"].reset_index(drop=True), days)
    it = items.reset_index(drop=True).assign(pos=days.searchsorted(td.fillna(days[-1])))[td.notna().to_numpy()]
    out = []
    for c, ev in d.groupby("code"):
        sub = it[it["code"] == c]
        if sub.empty:
            continue
        for ix, p in zip(ev.index, ev["pos"]):
            m = sub[(sub["pos"] >= p + lo) & (sub["pos"] <= p + hi)]
            if len(m):
                out.append(m.assign(ev=ix))
    return pd.concat(out, ignore_index=True) if out else items.iloc[:0].assign(ev=[], pos=[])


def cap_nearest(items: pd.DataFrame, close_time: pd.Series, source: str) -> pd.DataFrame:
    """每個事件、每個來源最多 CAP 則：取發布時間離事件日收盤最近的。close_time：ev -> t 收盤時間。"""
    dist = (items["time"] - items["ev"].map(close_time)).abs()
    keep = items.assign(_d=dist).sort_values(["ev", "_d"]).groupby("ev").head(CAP[source])
    return keep.drop(columns="_d")


def item_line(r) -> str:
    content = r.get("content")
    text = f"{r['text']}｜{content}" if isinstance(content, str) and content else r["text"]
    return f"[{r['item_id']}] {r['time']:%Y-%m-%d %H:%M} {SOURCE_NAME[r['source']]}｜{text}"


def user_message(ev, items: pd.DataFrame) -> str:
    """ev：事件列（code、name、sector、date、ar、kind）；items：這個事件的資料，依時間排序。"""
    t = ev["date"]
    if ev["kind"] == "event":
        move = f"t 當天{'大漲' if ev['ar'] > 0 else '大跌'}，超額報酬（扣掉大盤）{ev['ar'] * 100:+.1f}%。"
    else:
        move = f"t 當天沒有大漲跌，超額報酬 {ev['ar'] * 100:+.1f}%。請判斷這些資料裡和這家公司最相關的消息屬於哪一類。"
    head = (f"股票：{ev['code']} {ev['name']}（{ev['sector']}）\n"
            f"t = {t:%Y-%m-%d}，收盤 13:30、開盤 09:00。{move}\n"
            f"資料（t−6 收盤到 t+1 收盤，依時間排序）：")
    return head + "\n" + "\n".join(item_line(r) for _, r in items.iterrows())


def request_params(cfg, msg: str, cache_ttl=None) -> dict:
    cache_control = {"type": "ephemeral", **({"ttl": cache_ttl} if cache_ttl else {})}
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if cfg["llm"].get("effort"):
        output_config["effort"] = cfg["llm"]["effort"]
    return {"model": cfg["llm"]["model"], "max_tokens": 4096,
            "system": [{"type": "text", "text": SYSTEM, "cache_control": cache_control}],
            "messages": [{"role": "user", "content": msg}], "output_config": output_config}


def timing(cause_times: pd.Series, t: pd.Timestamp) -> str:
    """原因消息最早的發布時間相對於事件日：before（t 開盤前）、intraday（盤中）、after（收盤後）、none。"""
    if cause_times.empty:
        return "none"
    first = cause_times.min()
    if first < t + pd.Timedelta(hours=9):
        return "before"
    if first <= t + pd.Timedelta(hours=13, minutes=30):
        return "intraday"
    return "after"


def cohen_kappa(a, b) -> float:
    a, b = pd.Series(list(a)), pd.Series(list(b))
    cats = sorted(set(a) | set(b))
    po = float((a.values == b.values).mean())
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

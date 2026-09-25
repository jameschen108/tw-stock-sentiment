"""PTT Stock 板：原始 jsonl -> parquet，再挑出討論特定股票的文章與留言。"""
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CATEGORY_RE = re.compile(r"^(Re:\s*|Fw:\s*)?\[([^\]]+)\]")
# 引述行（※ 引述、: 開頭）不是作者自己的話；含「板規」「ex [標的]」等是發文範本
DROP_LINE_RE = re.compile(r"^(※\s*引述|[:：]|ex\s*\[)|板規|標題請使用以下格式")
TEMPLATE_RE = re.compile(r"[(（]例[^)）\n]*[)）]")   # 「(例 2330.TW 台積電)」

ART_SCHEMA = pa.schema([
    ("article_id", pa.string()), ("time", pa.timestamp("s")), ("author", pa.string()),
    ("title", pa.string()), ("category", pa.string()), ("is_reply", pa.bool_()),
    ("content", pa.string()), ("n_push", pa.int32()), ("n_boo", pa.int32()),
    ("n_arrow", pa.int32()),
])
COM_SCHEMA = pa.schema([
    ("article_id", pa.string()), ("idx", pa.int32()), ("time", pa.timestamp("s")),
    ("user", pa.string()), ("tag", pa.string()), ("content", pa.string()),
    ("lag_sec", pa.int64()),   # 留言時間 - 文章時間；PTT 留言時間沒有年份，年份是推出來的
])


def strip_quotes(text: str) -> str:
    keep = [ln for ln in (text or "").splitlines() if not DROP_LINE_RE.search(ln.strip())]
    return TEMPLATE_RE.sub("", "\n".join(keep)).strip()


def parse_category(title: str):
    m = CATEGORY_RE.match(title or "")
    return (m.group(2).strip() if m else None), bool(m and m.group(1))


def convert_year(src: Path, dst_dir: Path, chunk_rows: int = 500_000):
    """一個年度的 jsonl -> articles_{year}.parquet / comments_{year}.parquet"""
    year = re.search(r"(\d{4})", src.name).group(1)
    dst_dir.mkdir(parents=True, exist_ok=True)
    aw = pq.ParquetWriter(dst_dir / f"articles_{year}.parquet", ART_SCHEMA)
    cw = pq.ParquetWriter(dst_dir / f"comments_{year}.parquet", COM_SCHEMA)
    arts = {f.name: [] for f in ART_SCHEMA}
    coms = {f.name: [] for f in COM_SCHEMA}
    n_art = n_com = 0

    def flush(buf, writer, schema):
        if buf[schema.names[0]]:
            writer.write_table(pa.Table.from_pydict(buf, schema=schema))
            for v in buf.values():
                v.clear()

    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("date"):
            continue
        t = datetime.fromisoformat(r["date"])
        cat, is_reply = parse_category(r.get("title"))
        for k, v in [("article_id", r["article_id"]), ("time", t), ("author", r.get("author_id")),
                     ("title", r.get("title") or ""), ("category", cat), ("is_reply", is_reply),
                     ("content", strip_quotes(r.get("content"))), ("n_push", r.get("n_push") or 0),
                     ("n_boo", r.get("n_boo") or 0), ("n_arrow", r.get("n_arrow") or 0)]:
            arts[k].append(v)
        n_art += 1
        for c in r.get("comments") or []:
            if not c.get("time"):
                continue
            ct = datetime.fromisoformat(c["time"])
            for k, v in [("article_id", r["article_id"]), ("idx", c.get("index")), ("time", ct),
                         ("user", c.get("user_id")), ("tag", c.get("tag_en")),
                         ("content", c.get("content") or ""),
                         ("lag_sec", int((ct - t).total_seconds()))]:
                coms[k].append(v)
            n_com += 1
        if len(coms["article_id"]) >= chunk_rows:
            flush(coms, cw, COM_SCHEMA)
            flush(arts, aw, ART_SCHEMA)
    flush(arts, aw, ART_SCHEMA)
    flush(coms, cw, COM_SCHEMA)
    aw.close()
    cw.close()
    return n_art, n_com


def mention_pattern(ticker: str, aliases) -> str:
    """RE2 語法（pyarrow 用）：股號前後不能緊鄰數字，別名不分大小寫。"""
    alts = [f"(^|[^0-9]){ticker}([^0-9]|$)"] + [re.escape(a) for a in aliases]
    return "(?i)" + "|".join(alts)


def ticker_exclude(cfg, ticker) -> list:
    return (cfg.get("exclude") or {}).get(ticker, [])


def _mentions(arr, pat: str, exclude_pat: str | None):
    """先刪掉排除詞再比對，避免別名撞到其他公司（例如「長榮」撞「長榮航」）。"""
    if exclude_pat:
        arr = pc.replace_substring_regex(arr, exclude_pat, "")
    return pc.fill_null(pc.match_substring_regex(arr, pat), False)


def ticker_aliases(cfg, ticker) -> list:
    if ticker in (cfg.get("aliases") or {}):
        return cfg["aliases"][ticker]
    uni = pd.read_csv(Path(cfg["data_dir"]) / "universe_267.csv", dtype=str, encoding="utf-8-sig")
    row = uni[uni["ticker"] == ticker]
    return row["name_short"].tolist()


def select_texts(ptt_dir: Path, years, ticker: str, aliases, max_lag_days: int,
                 exclude=()) -> pd.DataFrame:
    """挑出討論這檔股票的文字。

    文章：標題提到它。只有內文提到的不算——實測大多是發文範本、融資融券表或順帶一提。
    留言：所在文章的標題提到它（整串都算），或留言本身提到它。
    回傳欄位：uid kind article_id time account text tag title category n_push n_boo
    """
    pat = mention_pattern(ticker, aliases)
    exclude_pat = "|".join(re.escape(e) for e in exclude) or None
    out = []
    for y in years:
        arts = pq.read_table(ptt_dir / f"articles_{y}.parquet")
        t_hit = _mentions(arts["title"], pat, exclude_pat)
        sel = arts.filter(t_hit).to_pandas()
        threads = arts.filter(t_hit)["article_id"].combine_chunks()
        titles = dict(zip(arts["article_id"].to_pylist(), arts["title"].to_pylist()))
        out.append(pd.DataFrame({
            "uid": "A:" + sel["article_id"], "kind": "article", "article_id": sel["article_id"],
            "time": sel["time"], "account": sel["author"],
            "text": sel["title"] + "\n" + sel["content"], "tag": None, "title": sel["title"],
            "category": sel["category"], "n_push": sel["n_push"], "n_boo": sel["n_boo"],
        }))

        coms = pq.read_table(ptt_dir / f"comments_{y}.parquet")
        in_thread = pc.is_in(coms["article_id"], value_set=threads)
        hit = _mentions(coms["content"], pat, exclude_pat)
        lag = coms["lag_sec"]
        lag_ok = pc.and_(pc.greater_equal(lag, -120),
                         pc.less_equal(lag, max_lag_days * 86400))
        c = coms.filter(pc.and_(pc.or_(in_thread, hit), lag_ok)).to_pandas()
        c_title = c["article_id"].map(titles)
        out.append(pd.DataFrame({
            "uid": "C:" + c["article_id"] + ":" + c["idx"].astype(str), "kind": "comment",
            "article_id": c["article_id"], "time": c["time"], "account": c["user"],
            "text": c["content"], "tag": c["tag"], "title": c_title,
            "category": c_title.map(lambda s: parse_category(s)[0]),
            "n_push": 0, "n_boo": 0,
        }))
    df = pd.concat(out, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)

"""[標的] 事件研究，第 1 步：把 [標的] 原文整理成事件表（不算任何報酬）。

    python scripts/07_target_events.py

需要：data/interim/target_posts.parquet（pttcc + pttweb 合併的 [標的] 原文）、
      data/prices/panel.parquet（scripts/build_panel.py）。
輸出 data/processed/target_events.parquet 與 output/target/coverage.csv（每一步排除多少、為什麼），
另外抽 200 則解析結果到 output/target/resolve_check.csv 供人工檢查。

事件：內文「分類」是多或空（標題方向相反就排除）、對到一檔台股、發文後有 entry 收盤。
split：dev = 2016–2023；final = 2024-01-01 以後（最終測試期，報酬要等預先登記後才算）。
2015 年的 [標的] 文幾乎沒有「分類」欄位，不收。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import events, panel as pnl  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402
from pttsent.sentiment.weak_labels import label_from_content  # noqa: E402

TARGET_LINE = re.compile(r"標的\s*[:：]\s*([^\n]{1,40})")
DEV_START, FINAL_START = pd.Timestamp("2016-01-01"), pd.Timestamp("2024-01-01")


def main():
    cfg = load_config()
    posts = pd.read_parquet(work_path(cfg, "interim", "target_posts.parquet"))
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                            columns=["date", "code", "name", "market"])
    days = pd.DatetimeIndex(sorted(panel.loc[panel["market"] == "TWSE", "date"].unique()))
    resolver = events.Resolver(events.listed_names(panel), {})

    df = posts[posts["time"] >= DEV_START].copy()
    df["split"] = np.where(df["time"] >= FINAL_START, "final", "dev")
    steps = [("[標的] 原文（2016 以後）", df)]

    df["dir_content"] = df["content"].map(label_from_content)
    df["dir_title"] = df["title"].map(events.title_direction)
    df = df[df["dir_content"].notna()]
    steps.append(("內文分類是多或空", df))
    df = df[df["dir_title"].isna() | (df["dir_title"] == df["dir_content"])].copy()
    df["direction"] = df["dir_content"]
    steps.append(("標題方向沒有衝突", df))

    res = [resolver.resolve(t, d) for t, d in zip(df["title"], df["time"])]
    df["code"], df["how"] = [r[0] for r in res], [r[1] for r in res]
    retry = df["code"].isna() & df["how"].isin(["no_tw_stock", "multi"])
    for i in df.index[retry]:                   # 標題對不到時，改看內文範本的「標的：」那一行
        m = TARGET_LINE.search(df.at[i, "content"] or "")
        if m:
            c, how = resolver.resolve(m.group(1), df.at[i, "time"])
            if c:
                df.at[i, "code"], df.at[i, "how"] = c, f"body_{how}"
    reasons = df.loc[df["code"].isna(), "how"].value_counts().to_dict()
    idx_posts = df[df["how"] == "index"].copy()      # 大盤／台指文：探索性分析用加權指數
    idx_posts["e"] = events.entry_positions(idx_posts["time"], days, cfg["cutoff"])
    idx_posts[idx_posts["e"] >= 0][["article_id", "time", "e", "split", "direction", "author", "title"]] \
        .to_parquet(work_path(cfg, "processed", "target_index_events.parquet"), index=False)
    df = df[df["code"].notna()].copy()
    steps.append(("對到一檔台股（含 ETF）", df))

    df["common"] = df["code"].str.fullmatch(pnl.COMMON.pattern)
    df["e"] = events.entry_positions(df["time"], days, cfg["cutoff"])
    df = df[df["e"] >= 0].copy()
    df["entry_date"] = days[df["e"]]
    listed = panel.drop_duplicates(["code", "date"]).sort_values("date")
    df = pd.merge_asof(df.sort_values("entry_date"), listed[["date", "code", "name", "market"]],
                       left_on="entry_date", right_on="date", by="code", direction="backward")
    df = df[df["market"].notna()].drop(columns="date")
    steps.append(("entry 時已上市櫃", df))

    # 重複：同一作者（沒有作者時用相同標題）5 個交易日內對同一檔、同方向，只留第一篇
    df = df.sort_values("time")
    who = df["author"].fillna("title:" + df["title"])
    df["_k"] = who + "|" + df["code"] + "|" + df["direction"]
    last_e = df.groupby("_k")["e"].shift(1)
    df = df[~(df["e"] - last_e <= 5)].drop(columns="_k")
    steps.append(("去掉 5 日內重複", df))
    prim = df[df["common"]]
    steps.append(("主要樣本：普通股", prim))

    def count(d):
        row = d.groupby("split").size().to_dict()
        if "direction" in d:
            row |= {f"{sp}_看空": int(((d["split"] == sp) & (d["direction"] == "bearish")).sum()) for sp in ("dev", "final")}
        return row
    cov = pd.DataFrame([{"step": s, **count(d)} for s, d in steps]).fillna(0)
    out_dir = output_path(cfg, "target", "x").parent
    cov.to_csv(out_dir / "coverage.csv", index=False)
    print(cov.to_string(index=False))
    print("\n對不到股票的原因：", reasons)
    print("解析方式：", df["how"].value_counts().to_dict())
    print("上市／上櫃：", prim.groupby(["split", "market"]).size().to_dict())

    keep = ["article_id", "time", "entry_date", "e", "split", "code", "name", "market", "common", "direction",
            "how", "author", "title", "n_push", "n_boo", "n_comments_web", "in_pttcc", "in_pttweb", "deleted"]
    out = work_path(cfg, "processed", "target_events.parquet")
    df[keep].to_parquet(out, index=False)
    df[df["split"] == "dev"].sample(200, random_state=0)[["title", "code", "name", "how", "direction"]] \
        .to_csv(out_dir / "resolve_check.csv", index=False)
    print(f"\n-> {out}\n-> {out_dir / 'coverage.csv'}\n-> {out_dir / 'resolve_check.csv'}（人工檢查用）")


if __name__ == "__main__":
    main()

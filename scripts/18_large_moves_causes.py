"""大漲跌的原因分類（docs/large_moves.md 第三部分）。只用開發期 2019–2023。

    python scripts/18_large_moves_causes.py build      # 組輸入、抽樣、產生人工驗證表（不花錢）
    python scripts/18_large_moves_causes.py tune       # 調整用的 30 個事件逐則呼叫，印出結果與費用
    python scripts/18_large_moves_causes.py tune --model claude-haiku-5-5 --effort low   # 改用別的模型或 effort（預設照 config.yaml）
    python scripts/18_large_moves_causes.py submit --effort low   # 提示詞 commit 之後：全部事件與 500 個控制日送 Batches API
    python scripts/18_large_moves_causes.py status
    python scripts/18_large_moves_causes.py collect    # 下載結果
    python scripts/18_large_moves_causes.py report     # 各類別比例、發布時間、和人工標註的一致率與 Cohen's κ

輸入、樣本、結果放在 data/interim/large_moves/causes/（含新聞內文，不進 git）。
人工驗證：validation.md 是 100 個事件的資料（看不到 LLM 的答案），在 docs/large_moves/causes_labels.csv 的 label 欄填類別。
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pttsent import causes as C  # noqa: E402
from pttsent import large_moves as lm  # noqa: E402
from pttsent import mops  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402
from pttsent.sentiment import llm_label as L  # noqa: E402

N_TUNE, N_VALID, N_CONTROL = 30, 100, 500
SEEDS = {"tune": 1, "valid": 2, "control": 3}
LABELS_CSV = ROOT / "docs" / "large_moves" / "causes_labels.csv"
REGISTERED = ["docs/large_moves.md", "src/pttsent/causes.py", "scripts/18_large_moves_causes.py"]


def out_dir(cfg) -> Path:
    return work_path(cfg, "interim", "large_moves", "causes", "x").parent


def load_rules():
    r = pd.read_csv(ROOT / "docs" / "large_moves" / "names_rules.csv", dtype={"code": str}, keep_default_na=False,
                    encoding="utf-8-sig")
    for c in ["ptt_name", "ptt_code", "news_title"]:
        r[c] = r[c].astype(str) == "True"
    return r


def matched_news(cfg, rules) -> pd.DataFrame:
    """提到每檔的鉅亨新聞（同 16_large_moves.py 的規則），保留 news_id、標題與內文。"""
    p = out_dir(cfg) / "news_content.parquet"
    if not p.exists():
        C.load_news_content(cfg["data_dir"]).to_parquet(p)
    news = pd.read_parquet(p)
    news = news[(news["time"] >= lm.DEV[0]) & (news["time"] < lm.DEV[1])].reset_index(drop=True)
    tags = pd.read_parquet(work_path(cfg, "interim", "large_moves", "news.parquet"), columns=["stock"])
    tags = tags["stock"].map(tuple).reindex(news["news_id"]).to_numpy()
    title = lm.norm_text(news["title"].fillna(""))
    out = []
    for r in rules.itertuples():
        hit = np.array([r.code in s if isinstance(s, tuple) else False for s in tags])
        if r.news_title:
            hit |= lm.mentions(title, r._replace(ptt_code=False, ptt_name=True))
        sub = news[hit]
        out.append(pd.DataFrame({"code": r.code, "time": sub["time"], "text": sub["title"],
                                 "content": sub["content"], "ref": sub["news_id"]}))
    return pd.concat(out, ignore_index=True)


def matched_ptt(cfg, rules) -> pd.DataFrame:
    """標題提到每檔的 PTT 文章（不含回文、轉文）。"""
    out = []
    for y in range(2019, 2024):
        a = pq.read_table(work_path(cfg, "interim", "ptt", f"articles_{y}.parquet"),
                          columns=["article_id", "time", "title", "is_reply"]).to_pandas()
        a = a[~a["is_reply"] & ~a["title"].fillna("").str.startswith("Fw:")]
        title = lm.norm_text(a["title"].fillna(""))
        for r in rules.itertuples():
            hit = lm.mentions(title, r)
            out.append(pd.DataFrame({"code": r.code, "time": a.loc[hit, "time"], "text": a.loc[hit, "title"],
                                     "ref": a.loc[hit, "article_id"]}))
    return pd.concat(out, ignore_index=True)


def cmd_build(cfg):
    days = pd.DatetimeIndex(sorted(pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "market"])
                                   .query("market == 'TWSE'")["date"].unique()))
    w = pd.read_parquet(work_path(cfg, "interim", "large_moves", "windows.parquet"))
    u = lm.universe(cfg["data_dir"]).set_index("code")
    rules = load_rules()
    src = {"news": matched_news(cfg, rules)}
    ann = mops.load(work_path(cfg, "mops", "t05st01", "x").parent)
    ann = ann[(ann["time"] >= lm.DEV[0]) & (ann["time"] < lm.DEV[1])]
    src["mops"] = pd.DataFrame({"code": ann["code"], "time": ann["time"], "text": ann["subject"],
                                "ref": ann["code"] + "_" + ann["time"].dt.strftime("%Y%m%d%H%M%S")})
    src["ptt"] = matched_ptt(cfg, rules)

    # 控制日：t−1 收盤到 t+1 收盤之間有鉅亨新聞或重大訊息的控制日，隨機 500 個
    ctrl = w[w["control"]]
    has = set()
    for s in ("news", "mops"):
        has |= set(C.window_items(src[s], days, ctrl, 0, 1)["ev"])
    ctrl = ctrl.loc[sorted(has)].sample(N_CONTROL, random_state=SEEDS["control"])
    ev = pd.concat([w[w["main"]].assign(kind="event"), ctrl.assign(kind="control")])
    ev = ev.assign(name=ev["code"].map(u["name"]), sector=ev["code"].map(u["sector"]),
                   close=ev["date"] + pd.Timedelta(hours=13, minutes=30))

    items = []
    for s, df in src.items():
        it = C.window_items(df, days, ev, -5, 1)
        items.append(C.cap_nearest(it, ev["close"], s).assign(source=s))
    items = pd.concat(items, ignore_index=True).sort_values(["ev", "time", "source"])
    prefix = {"news": "N", "mops": "M", "ptt": "P"}
    items["item_id"] = items["source"].map(prefix) + (items.groupby(["ev", "source"]).cumcount() + 1).astype(str)

    n_info = items[items["source"] != "ptt"].groupby("ev").size()
    ev["n_news"] = items[items["source"] == "news"].groupby("ev").size().reindex(ev.index, fill_value=0)
    ev["n_mops"] = items[items["source"] == "mops"].groupby("ev").size().reindex(ev.index, fill_value=0)
    ev["n_ptt"] = items[items["source"] == "ptt"].groupby("ev").size().reindex(ev.index, fill_value=0)
    ev["no_news"] = ~ev.index.isin(n_info.index)
    ev["ev_id"] = [f"{k[0]}{c}_{d:%Y%m%d}" for k, c, d in zip(ev["kind"], ev["code"], ev["date"])]
    by_ev = dict(tuple(items.groupby("ev")))
    ev["msg"] = [C.user_message(r, by_ev[i]) if i in by_ev and not r["no_news"] else "" for i, r in ev.iterrows()]

    # 調整用 30 個、驗證用 100 個：只從有資料、會送 LLM 的主要事件抽，漲跌各半，兩組不重疊
    pool = ev[(ev["kind"] == "event") & ~ev["no_news"]]
    tune = pd.concat([g.sample(N_TUNE // 2, random_state=SEEDS["tune"]) for _, g in pool.groupby("up")])
    rest = pool.drop(tune.index)
    valid = pd.concat([g.sample(N_VALID // 2, random_state=SEEDS["valid"]) for _, g in rest.groupby("up")])
    ev["set"] = "main"
    ev.loc[tune.index, "set"] = "tune"
    ev.loc[valid.index, "set"] = "valid"

    d = out_dir(cfg)
    keep = ["ev_id", "kind", "set", "code", "name", "sector", "date", "pos", "ar", "z", "up", "cluster", "locked",
            "n_news", "n_mops", "n_ptt", "no_news", "msg"]
    ev[keep].reset_index(drop=True).to_parquet(d / "events.parquet")
    items.assign(ev_id=items["ev"].map(ev["ev_id"])).drop(columns=["ev"]).to_parquet(d / "items.parquet")

    # 人工驗證表：只有資料，沒有 LLM 的答案；順序打亂，不顯示漲跌以外的提示
    v = ev.loc[valid.index].sample(frac=1, random_state=SEEDS["valid"])
    md = ["# 大漲跌原因：人工驗證（100 個事件）", "",
          "每個事件選一個類別，填在 docs/large_moves/causes_labels.csv 的 label 欄。類別：" + "、".join(C.CATEGORIES) + "。",
          "類別的定義和 LLM 拿到的一樣，見 src/pttsent/causes.py 的 SYSTEM。", ""]
    for i, r in v.iterrows():
        md += [f"## {r['ev_id']}", "", "```", r["msg"], "```", ""]
    (d / "validation.md").write_text("\n".join(md), encoding="utf-8")
    if not LABELS_CSV.exists():
        v[["ev_id"]].assign(label="", note="").to_csv(LABELS_CSV, index=False, encoding="utf-8-sig")

    sent = ev[ev["msg"] != ""]
    chars = sent["msg"].str.len() + len(C.SYSTEM)
    print(f"主要事件 {int((ev['kind'] == 'event').sum())}（沒有鉅亨也沒有重大訊息 {int(ev.loc[ev['kind'] == 'event', 'no_news'].sum())}），"
          f"控制日 {int((ev['kind'] == 'control').sum())}；要送 LLM {len(sent)} 個")
    print(f"每個事件：鉅亨 {sent['n_news'].mean():.1f}、重大訊息 {sent['n_mops'].mean():.1f}、PTT {sent['n_ptt'].mean():.1f} 則；"
          f"輸入 {chars.mean():,.0f} 字（中位數 {chars.median():,.0f}、最多 {chars.max():,}）")
    print(f"-> {d}；人工驗證表 {d / 'validation.md'}、{LABELS_CSV}")


FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")   # 支援伺服器端備援的模型


def with_llm(cfg, model=None, effort=None) -> dict:
    """指令列指定的模型、effort 蓋過 config.yaml。"""
    llm = {**cfg["llm"], **({"model": model} if model else {}), **({"effort": effort} if effort else {})}
    return {**cfg, "llm": llm}


def run_sync(cfg, ev: pd.DataFrame) -> pd.DataFrame:
    """逐則呼叫（調整用）。支援的模型被拒答時，由伺服器改用其他模型。"""
    import anthropic
    client = L._client()
    rows = []
    for _, r in ev.iterrows():
        params = C.request_params(cfg, r["msg"])
        try:
            if cfg["llm"]["model"] in FALLBACK_MODELS:
                msg = client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                msg = client.messages.create(**params)
            rows.append({"ev_id": r["ev_id"], **L._parse_message(msg)})
        except anthropic.APIStatusError as e:
            rows.append({"ev_id": r["ev_id"], "status": f"http_{e.status_code}"})
        except anthropic.APIConnectionError:
            rows.append({"ev_id": r["ev_id"], "status": "connection_error"})
    return ev.merge(pd.DataFrame(rows), on="ev_id")


def cmd_tune(cfg):
    d = out_dir(cfg)
    ev = pd.read_parquet(d / "events.parquet")
    res = run_sync(cfg, ev[ev["set"] == "tune"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{cfg['llm']['model']}_{cfg['llm'].get('effort') or 'default'}"
    res.assign(model=cfg["llm"]["model"], effort=cfg["llm"].get("effort")).to_parquet(d / f"tune_{tag}_{stamp}.parquet")
    print("status", res["status"].value_counts().to_dict(), "usage", L.usage_summary(res))
    for _, r in res.iterrows():
        print(f"\n=== {r['ev_id']} ar {r['ar'] * 100:+.1f}% -> {r.get('category')} {r.get('cause_ids')} "
              f"{r.get('confidence')}：{r.get('reason')}")


def committed(paths) -> bool:
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def cmd_submit(cfg):
    if not committed(REGISTERED):
        sys.exit("提示詞或程式有沒 commit 的修改；先 commit 再送")
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    d = out_dir(cfg)
    log_path = d / "batches.json"
    log = json.load(open(log_path)) if log_path.exists() else {"batches": []}
    sent = {i for b in log["batches"] for i in b["ev_ids"]}
    ev = pd.read_parquet(d / "events.parquet")
    todo = ev[(ev["msg"] != "") & (ev["set"] != "tune") & ~ev["ev_id"].isin(sent)]
    if todo.empty:
        print("都送過了")
        return
    reqs = [Request(custom_id=r["ev_id"], params=MessageCreateParamsNonStreaming(**C.request_params(cfg, r["msg"], "1h")))
            for _, r in todo.iterrows()]
    batch = L._client().messages.batches.create(requests=reqs)
    log["batches"].append({"batch_id": batch.id, "n": len(todo), "model": cfg["llm"]["model"],
                           "effort": cfg["llm"].get("effort"),
                           "created": datetime.now().isoformat(), "ev_ids": todo["ev_id"].tolist()})
    json.dump(log, open(log_path, "w"), indent=2)
    print(f"已送出 batch {batch.id}（{len(todo):,} 個）")


def cmd_status(cfg):
    log = json.load(open(out_dir(cfg) / "batches.json"))
    for b in log["batches"]:
        s = L.batch_status(b["batch_id"])
        print(b["batch_id"], b["n"], s.processing_status, s.request_counts)


def cmd_collect(cfg):
    d = out_dir(cfg)
    log = json.load(open(d / "batches.json"))
    client = L._client()
    rows = []
    for b in log["batches"]:
        for res in client.messages.batches.results(b["batch_id"]):
            if res.result.type == "succeeded":
                rows.append({"ev_id": res.custom_id, **L._parse_message(res.result.message)})
            else:
                rows.append({"ev_id": res.custom_id, "status": res.result.type})
    out = pd.DataFrame(rows)
    out.to_parquet(d / "results.parquet")
    print("status", out["status"].value_counts().to_dict(), "usage", L.usage_summary(out, batch=True))


def cmd_report(cfg):
    d = out_dir(cfg)
    ev = pd.read_parquet(d / "events.parquet")
    items = pd.read_parquet(d / "items.parquet")
    res = pd.read_parquet(d / "results.parquet")
    ev = ev.merge(res[["ev_id", "status", "category", "cause_ids", "confidence", "reason"]], on="ev_id", how="left")
    empty = ev["cause_ids"].map(lambda x: x is None or len(x) == 0)
    ev.loc[(ev["category"] == "其他") & empty, "category"] = C.NOT_FOUND
    ev.loc[ev["no_news"], "category"] = C.NO_NEWS
    t_of = items.set_index(["ev_id", "item_id"])["time"]
    ev["timing"] = [C.timing(pd.Series([t_of.get((e, i)) for i in (ids if ids is not None else [])], dtype="datetime64[ns]").dropna(), t)
                    for e, ids, t in zip(ev["ev_id"], ev["cause_ids"], ev["date"])]
    ev.drop(columns="msg").to_parquet(output_path(cfg, "large_moves", "causes.parquet"))

    cats = C.CATEGORIES + [C.NOT_FOUND, C.NO_NEWS]
    main = ev[(ev["kind"] == "event") & (ev["set"] != "tune")]
    tabs = {
        "大漲／大跌": pd.crosstab(main["category"], main["up"].map({True: "大漲", False: "大跌"}), normalize="columns"),
        "事件／控制日": pd.crosstab(ev.loc[ev["set"] != "tune", "category"], ev.loc[ev["set"] != "tune", "kind"], normalize="columns"),
        "大小": pd.crosstab(main["category"], np.where(main["z"].abs() >= 5, "5σ 以上", "3–5σ"), normalize="columns"),
        "群聚日": pd.crosstab(main["category"], main["cluster"].map({True: "群聚日", False: "非群聚日"}), normalize="columns"),
        "產業": pd.crosstab(main["sector"], main["category"], normalize="index"),
        "原因發布時間": pd.crosstab(main["category"], main["timing"], normalize="index"),
    }
    with open(output_path(cfg, "large_moves", "causes_tables.txt"), "w") as f:
        for k, t in tabs.items():
            if k != "產業":
                t = t.reindex([c for c in cats if c in t.index])
            f.write(f"## {k}\n{t.round(3).to_string()}\n\n")
            print(f"## {k}\n{t.round(3).to_string()}\n")

    lab = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    lab = lab[lab["label"] != ""].merge(ev[["ev_id", "category"]], on="ev_id")
    if len(lab):
        agree = float((lab["label"] == lab["category"]).mean())
        print(f"人工驗證 {len(lab)} 個：一致率 {agree:.1%}，Cohen's κ = {C.cohen_kappa(lab['label'], lab['category']):.3f}")
        print(pd.crosstab(lab["label"], lab["category"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "tune", "submit", "status", "collect", "report"])
    ap.add_argument("--model", help="蓋過 config.yaml 的 llm.model（tune、submit）")
    ap.add_argument("--effort", help="蓋過 config.yaml 的 llm.effort（tune、submit）")
    a = ap.parse_args()
    cfg = with_llm(load_config(), a.model, a.effort)
    {"build": cmd_build, "tune": cmd_tune, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect,
     "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()

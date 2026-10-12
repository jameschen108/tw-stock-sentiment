"""開盤前情緒 → 盤中報酬（D2）的預先登記檢定（見 docs/prereg/PREREGISTRATION_DIRECTION.md）。設定全部寫死。

    python scripts/20_direction_prereg.py power        # 檢定力：只用開發期的結果
    python scripts/20_direction_prereg.py reproduce    # 用這支程式重算開發期，確認和 19 的 gapcheck（C1）相同
    python scripts/20_direction_prereg.py score A      # 抽出評估期的開盤前文字、BERT 打分數（可中斷後接續；不看報酬）
    python scripts/20_direction_prereg.py check A      # 只檢查資料涵蓋（每月的樣本數、缺值），不跑回歸
    python scripts/20_direction_prereg.py final A      # E1：事件日 2024-01-01 .. 2026-09-24，只跑一次
    python scripts/20_direction_prereg.py final B      # 同上：2026-10-01 .. 2029-09-30，期間結束、資料補齊後只跑一次

E1：oc(t) ~ sent + gap_z + 控制變數 + 日期固定效果，sent 的係數 > 0，單尾 α = 0.05。
文字比對、情緒分數、每日情緒、事件定義都沿用開發期已 commit 的程式（direction.py、large_moves.py）。
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pttsent import direction as D  # noqa: E402
from pttsent import large_moves as lm  # noqa: E402
from pttsent import mops  # noqa: E402
from pttsent import volume as v  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402

PERIODS = {"A": ("2024-01-01", "2026-09-24"), "B": ("2026-10-01", "2029-09-30")}
DEV = (lm.EVENT_START, lm.EVENT_END)   # 只給 reproduce 用
MOPS_DIR = {"A": "t05st01", "B": "t05st01_2029", "dev": "t05st01"}
DEV_T, DEV_DAYS = 2.86, 1161           # 開發期 C1：D2 加跳空的 t 值、交易日數
ALPHA, CHUNK = 0.05, 20_000
BERT_DIR = ROOT / "data" / "models" / "sentiment_bert_llm_pooled"
BERT_SHA256 = "a7d7d0f5d5325394003dd9b399016efa3d20e8745f836a7b2f8553912b14d86d"
X = ["sent", "gap_z", "ar1_z", "ar5_z", "ptt_abn_pre_early", "news_log_preopen", "mops_log_preopen"]
REGISTERED = ["docs/prereg/PREREGISTRATION_DIRECTION.md", "scripts/20_direction_prereg.py",
              "src/pttsent/direction.py", "src/pttsent/large_moves.py", "src/pttsent/mops.py",
              "docs/large_moves/names_rules.csv"]


def committed(paths) -> bool:
    if not all((ROOT / f).exists() for f in paths):
        return False
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def load_rules():
    r = pd.read_csv(ROOT / "docs" / "large_moves" / "names_rules.csv", dtype={"code": str}, keep_default_na=False,
                    encoding="utf-8-sig")
    for c in ["ptt_name", "ptt_code", "news_title"]:
        r[c] = r[c].astype(str) == "True"
    return r


def market_days(cfg) -> pd.DatetimeIndex:
    m = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "market"])
    return pd.DatetimeIndex(sorted(m.loc[m["market"] == "TWSE", "date"].unique()))


def text_range(days: pd.DatetimeIndex, first: str, last: str) -> tuple[str, str]:
    """文字資料的起訖：最早到基準期第一天（t−26 收盤），最晚到最後一個事件日的 08:30。"""
    start = days[max(days.searchsorted(pd.Timestamp(first)) - lm.BASE[0] - 2, 0)]
    return str(start.date()), str(pd.Timestamp(last) + pd.Timedelta(hours=lm.TRIAL[0], minutes=lm.TRIAL[1]))


def work_dir(cfg, period) -> Path:
    if period == "dev":   # 開發期的分數由 19_large_moves_direction.py 打好了
        return work_path(cfg, "interim", "large_moves", "direction", "x").parent
    return work_path(cfg, "interim", "large_moves", "direction_prereg", period, "x").parent


def cmd_score(cfg, period):
    """評估期的開盤前文字與 BERT 分數。只處理文字，不碰報酬。"""
    from pttsent.sentiment import bert
    if hashlib.sha256((BERT_DIR / "model.safetensors").read_bytes()).hexdigest() != BERT_SHA256:
        sys.exit("BERT 模型和登記時的不同（sha256 不符）")
    days = market_days(cfg)
    first, last = PERIODS[period]
    start, end = text_range(days, first, last)
    d = work_dir(cfg, period)
    if not (d / "texts.parquet").exists():
        t = D.preopen_texts(work_path(cfg, "interim", "ptt", "x").parent, range(int(start[:4]), int(last[:4]) + 1),
                            load_rules(), days, cfg["sentiment"]["max_comment_lag_days"], start, end)
        t.to_parquet(d / "texts.parquet")
    t = pd.read_parquet(d / "texts.parquet", columns=["kind", "text", "title"])
    u = t.drop_duplicates(["text", "title"]).reset_index(drop=True)
    u.to_parquet(d / "unique_texts.parquet")
    model, tok = bert.load(BERT_DIR)
    for i in range(0, len(u), CHUNK):
        f = d / f"scores_{i // CHUNK:04d}.parquet"
        if f.exists():
            continue
        part = u.iloc[i:i + CHUNK]
        s = bert.score(model, tok, part["text"].tolist(), bert.titles_of(part).tolist())
        pd.DataFrame({"row": np.arange(i, i + len(part)), "score": s}).to_parquet(f)
        print(f"{min(i + CHUNK, len(u)):,} / {len(u):,}", flush=True)


def build(cfg, period) -> pd.DataFrame:
    """評估期內合格、不是後續事件的股票日，加上 E1 用到的變數。"""
    first, last = DEV if period == "dev" else PERIODS[period]
    days = market_days(cfg)
    start, end = text_range(days, first, last)
    u = lm.universe(cfg["data_dir"])
    rules = load_rules()
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                            columns=["date", "code", "market", "close", "volume", "base", "ret", "mkt_ret", "oc", "gap",
                                     "limit_up_open", "limit_down_open", "limit_up_close", "limit_down_close"])
    panel = panel[panel["code"].isin(set(u["code"])) & (panel["date"] <= last)]
    full = lm.stock_days(panel, days, v.load_disposition(cfg["data_dir"]), u["code"]).sort_values(["code", "date"])
    g = full.groupby("code", sort=False)["ar"]
    full["ar1_z"] = g.shift(1) / full["sigma"]
    full["ar5_z"] = sum(g.shift(j) for j in range(1, 6)) / full["sigma"]
    full["gap_z"] = full["gap"].where(full["gap"].abs() <= 0.105) / full["sigma"]
    full["open_locked"] = full["limit_up_open"].fillna(False) | full["limit_down_open"].fillna(False)
    d = full[(full["date"] >= first) & (full["date"] <= last) & full["eligible"] & ~full["followon"]].copy()

    ptt = lm.ptt_items(work_path(cfg, "interim", "ptt", "x").parent, range(int(start[:4]), int(last[:4]) + 1), rules,
                       cfg["sentiment"]["max_comment_lag_days"], end)
    ptt = ptt[ptt["time"] >= start]
    news = pd.read_parquet(work_path(cfg, "interim", "large_moves", "news.parquet"))
    news["stock"] = news["stock"].map(tuple)
    news = lm.news_items(news[(news["time"] >= start) & (news["time"] < end)], rules)
    ann = mops.load(work_path(cfg, "mops", MOPS_DIR[period], "x").parent)
    ann = ann[(ann["time"] >= start) & (ann["time"] < end)][["code", "time"]]
    d = lm.add_windows(d, ptt, days, "ptt", start=start)
    d = lm.add_windows(d, news, days, "news", start=start)
    d = lm.add_windows(d, ann, days, "mops", start=start)
    d["news_log_preopen"] = np.log1p(d["news_preopen"])
    d["mops_log_preopen"] = np.log1p(d["mops_preopen"])
    split = lm.preopen_split(d, ptt, days)
    d["ptt_abn_pre_early"] = np.log1p(split["pre_early"]) - d["ptt_base"]

    wd = work_dir(cfg, period)
    t = pd.read_parquet(wd / "texts.parquet")
    uq = pd.read_parquet(wd / "unique_texts.parquet")
    s = pd.concat([pd.read_parquet(f) for f in sorted(wd.glob("scores_*.parquet"))]).set_index("row")["score"]
    if len(s) != len(uq):
        sys.exit(f"分數只有 {len(s):,} / {len(uq):,} 則，先跑完 score {period}")
    uq["score"] = s.reindex(range(len(uq))).to_numpy()
    sent = D.daily_sent(t.merge(uq[["text", "title", "score"]], on=["text", "title"], how="left"))
    d = d.merge(sent, on=["code", "pos"], how="left")
    sample = (~d["open_locked"]) & (d["ptt_abn_pre_early"] >= D.D2_ABN_MIN) & d["sent"].notna()
    return d.assign(in_sample=sample)


def cmd_power(cfg, period):
    days = market_days(cfg)
    z = norm.ppf(1 - ALPHA)
    for k, (a, b) in PERIODS.items():
        n = int(((days >= a) & (days <= b)).sum()) if k == "A" else 3 * 245   # B 還沒有交易日曆，以每年 245 天估
        for share in (1.0, 0.5):
            t = DEV_T * share * np.sqrt(n / DEV_DAYS)
            print(f"{k}（{n} 天）效果 × {share}：預期 t = {t:.2f}，檢定力 {norm.cdf(t - z):.0%}")


def cmd_reproduce(cfg, period):
    d = build(cfg, "dev")
    f = lm.fe_ols(d[d["in_sample"]], "oc", X, ["date"])
    print(f"開發期 E1：係數 {f.params['sent']:.5f}，t = {f.tvalues['sent']:.3f}，n = {int(f.nobs)}"
          f"（19 的 gapcheck C1：0.0048，t = 2.86，n = 6,067）")


def cmd_check(cfg, period):
    d = build(cfg, period)
    s = d[d["in_sample"]]
    print(f"合格股票日 {len(d):,}，E1 樣本 {len(s):,}（{s['code'].nunique()} 檔）")
    print(s.groupby(s["date"].dt.to_period("M")).size().to_string())
    for x in X + ["oc"]:
        print(f"{x}: 缺值 {s[x].isna().mean():.4f}")


def cmd_final(cfg, period):
    first, last = PERIODS[period]
    out = output_path(cfg, "large_moves", f"prereg_direction_{period}.json")
    if not committed(REGISTERED):
        sys.exit("要在預先登記 commit 之後才能跑，這些檔案不存在或有未 commit 的改動：" + "、".join(REGISTERED))
    if hashlib.sha256((BERT_DIR / "model.safetensors").read_bytes()).hexdigest() != BERT_SHA256:
        sys.exit("BERT 模型和登記時的不同（sha256 不符）")
    if date.today() <= date.fromisoformat(last):
        sys.exit(f"評估期到 {last} 才結束，現在不能跑")
    if out.exists():
        sys.exit(f"{out} 已經存在：這個檢定已經跑過，不能再跑")
    d = build(cfg, period)
    s = d[d["in_sample"]]
    f = lm.fe_ols(s, "oc", X, ["date"])
    t = f.tvalues["sent"]
    res = {"period": period, "window": [first, last],
           "E1": {"coef": f.params["sent"], "se": f.bse["sent"], "t": t, "p": 1 - norm.cdf(t), "n": int(f.nobs),
                  "n_stocks": int(s["code"].nunique()), "effect_1sd_bp": 1e4 * f.params["sent"] * s["sent"].std(),
                  "pass": bool(1 - norm.cdf(t) < ALPHA)},
           "gap_coef": f.params["gap_z"], "gap_t": f.tvalues["gap_z"]}
    # 另外報告：照 sent 的正負號開盤進場、收盤出場的平均毛報酬與扣成本後
    c = cfg["costs"]
    tr = s.dropna(subset=["oc"])
    tr = tr[tr["sent"] != 0]
    gross = np.sign(tr["sent"]) * np.expm1(tr["oc"])
    res["trade"] = {"n": len(gross), "gross": float(gross.mean()), "t": float(gross.mean() / gross.std() * np.sqrt(len(gross))),
                    "net_stock": float(gross.mean() - (2 * c["fee_rate"] * c["fee_discount"] + c["daytrade_tax_rate"])),
                    "net_futures": float(gross.mean() - 2 * (c["futures"]["fee_rate"] + c["futures"]["tax_rate"]))}
    res["run_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(res, ensure_ascii=False, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["power", "reproduce", "score", "check", "final"])
    ap.add_argument("period", nargs="?", choices=list(PERIODS))
    a = ap.parse_args()
    if a.cmd not in ("power", "reproduce") and not a.period:
        ap.error("score / check / final 要指定 A 或 B")
    cfg = load_config()
    {"power": cmd_power, "reproduce": cmd_reproduce, "score": cmd_score, "check": cmd_check,
     "final": cmd_final}[a.cmd](cfg, a.period)


if __name__ == "__main__":
    main()

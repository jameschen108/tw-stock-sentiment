"""開盤前的情緒能不能預告漲跌方向（docs/large_moves_direction.md）。只用開發期 2019–2023。

    python scripts/19_large_moves_direction.py texts     # 抽出每檔在 t−1 收盤到 t 08:30 之間的 PTT 文字
    python scripts/19_large_moves_direction.py score     # BERT 打情緒分數（Apple GPU 約 1–2 小時，可中斷後接續）
    python scripts/19_large_moves_direction.py analyze   # D1、D2 與次要結果；計畫和程式都 commit 之後才會跑
    python scripts/19_large_moves_direction.py gapcheck  # 事後檢查：控制開盤跳空（C1–C3）

中間產物在 data/interim/large_moves/direction/，結果在 output/large_moves/direction_*.csv。
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pttsent import direction as D  # noqa: E402
from pttsent import large_moves as lm  # noqa: E402
from pttsent import mops  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402

FDR = 0.10
CHUNK = 20_000
BERT_DIR = ROOT / "data" / "models" / "sentiment_bert_llm_pooled"
REGISTERED = ["docs/large_moves_direction.md", "src/pttsent/direction.py", "scripts/19_large_moves_direction.py"]
CONTROLS = ["ar1_z", "ar5_z", "ptt_abn_pre_early", "news_log_preopen", "mops_log_preopen"]


def out_dir(cfg) -> Path:
    return work_path(cfg, "interim", "large_moves", "direction", "x").parent


def market_days(cfg) -> pd.DatetimeIndex:
    m = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "market"])
    return pd.DatetimeIndex(sorted(m.loc[m["market"] == "TWSE", "date"].unique()))


def load_rules():
    r = pd.read_csv(ROOT / "docs" / "large_moves" / "names_rules.csv", dtype={"code": str}, keep_default_na=False,
                    encoding="utf-8-sig")
    for c in ["ptt_name", "ptt_code", "news_title"]:
        r[c] = r[c].astype(str) == "True"
    return r


def cmd_texts(cfg):
    days = market_days(cfg)
    t = D.preopen_texts(work_path(cfg, "interim", "ptt", "x").parent, range(2019, 2024), load_rules(), days,
                        cfg["sentiment"]["max_comment_lag_days"], "2019-01-01", lm.DEV[1])
    t.to_parquet(out_dir(cfg) / "texts.parquet")
    u = t.drop_duplicates(["text", "title"])
    print(f"{len(t):,} 則（{t['kind'].value_counts().to_dict()}），不重複的 {len(u):,} 則，{t['code'].nunique()} 檔")


def cmd_score(cfg):
    """不重複的（text, title）分塊打分數，每塊存一個檔，中斷後重跑會跳過已完成的塊。"""
    from pttsent.sentiment import bert
    d = out_dir(cfg)
    t = pd.read_parquet(d / "texts.parquet", columns=["kind", "text", "title"])
    u = t.drop_duplicates(["text", "title"]).reset_index(drop=True)
    u.to_parquet(d / "unique_texts.parquet")
    model, tok = bert.load(BERT_DIR)
    chunks = range(0, len(u), CHUNK)
    for i in chunks:
        f = d / f"scores_{i // CHUNK:04d}.parquet"
        if f.exists():
            continue
        part = u.iloc[i:i + CHUNK]
        s = bert.score(model, tok, part["text"].tolist(), bert.titles_of(part).tolist())
        pd.DataFrame({"row": np.arange(i, i + len(part)), "score": s}).to_parquet(f)
        print(f"{min(i + CHUNK, len(u)):,} / {len(u):,}", flush=True)


def load_scored(cfg) -> pd.DataFrame:
    d = out_dir(cfg)
    t = pd.read_parquet(d / "texts.parquet")
    u = pd.read_parquet(d / "unique_texts.parquet")
    s = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("scores_*.parquet"))]).set_index("row")["score"]
    if len(s) != len(u):
        sys.exit(f"分數只有 {len(s):,} / {len(u):,} 則，先跑完 score")
    u["score"] = s.reindex(range(len(u))).to_numpy()
    return t.merge(u[["text", "title", "score"]], on=["text", "title"], how="left")


def committed(paths) -> bool:
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def build(cfg) -> pd.DataFrame:
    """事件期間內每個合格股票日：sent、控制變數、oc 報酬。"""
    days = market_days(cfg)
    w = pd.read_parquet(work_path(cfg, "interim", "large_moves", "windows.parquet"))
    full = pd.read_parquet(work_path(cfg, "interim", "large_moves", "days.parquet")).sort_values(["code", "date"])
    g = full.groupby("code", sort=False)["ar"]
    full["ar1"] = g.shift(1)
    full["ar5"] = sum(g.shift(j) for j in range(1, 6))
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                            columns=["date", "code", "oc", "gap", "limit_up_open", "limit_down_open"])
    w = w.merge(full[["code", "date", "ar1", "ar5"]], on=["code", "date"], how="left")
    w = w.merge(panel.drop_duplicates(["code", "date"]), on=["code", "date"], how="left")
    w["ar1_z"] = w["ar1"] / w["sigma"]
    w["ar5_z"] = w["ar5"] / w["sigma"]
    w["gap_z"] = w["gap"].where(w["gap"].abs() <= 0.105) / w["sigma"]

    split = lm.preopen_split(w, pd.read_parquet(work_path(cfg, "interim", "large_moves", "ptt_items.parquet")), days)
    w["ptt_abn_pre_early"] = np.log1p(split["pre_early"]) - w["ptt_base"]
    ann = mops.load(work_path(cfg, "mops", "t05st01", "x").parent)
    ann = ann[(ann["time"] >= "2019-01-01") & (ann["time"] < lm.DEV[1])][["code", "time"]]
    w = lm.add_windows(w, ann, days, "mops", start="2019-01-01", end=lm.DEV[1])
    w["mops_log_preopen"] = np.log1p(w["mops_preopen"])

    sent = D.daily_sent(load_scored(cfg))
    w = w.merge(sent, on=["code", "pos"], how="left")
    w["n_accounts"] = w["n_accounts"].fillna(0)
    w["year"] = w["date"].dt.year
    w["open_locked"] = w["limit_up_open"].fillna(False) | w["limit_down_open"].fillna(False)
    return w


def one_sided(f, x) -> dict:
    t = f.tvalues[x]
    return {"coef": f.params[x], "se": f.bse[x], "t": t, "p": 1 - norm.cdf(t), "n": int(f.nobs)}


def cmd_analyze(cfg):
    if not committed(REGISTERED):
        sys.exit("計畫或程式有沒 commit 的修改；先 commit 再跑")
    w = build(cfg)
    ev = w[w["main"]].assign(y=lambda x: x["up"].astype(float))
    d2 = w[w["eligible"] & ~w["followon"] & ~w["open_locked"]]

    f1 = lm.fe_ols(ev.dropna(subset=["sent"]), "y", ["sent"] + CONTROLS, ["year"])
    f2 = lm.fe_ols(d2[d2["ptt_abn_pre_early"] >= D.D2_ABN_MIN].dropna(subset=["sent"]), "oc", ["sent"] + CONTROLS, ["date"])
    prim = pd.DataFrame([{"test": "D1", "desc": "sent → 大漲（大漲跌日）", **one_sided(f1, "sent")},
                         {"test": "D2", "desc": "sent → 盤中報酬（開盤前討論異常熱）", **one_sided(f2, "sent")}])
    rej, p_fdr, _, _ = multipletests(prim["p"], alpha=FDR, method="fdr_bh")
    prim["p_fdr"], prim["pass"] = p_fdr, rej
    prim.to_csv(output_path(cfg, "large_moves", "direction_primary.csv"), index=False, encoding="utf-8-sig")
    print("主要檢定：\n" + prim.round(4).to_string(index=False))

    # 次要結果（只描述）
    sec = []
    e10 = ev[ev["n_accounts"] >= 10].dropna(subset=["sent"])
    sec.append({"結果": "D1，帳號 ≥ 10", **one_sided(lm.fe_ols(e10, "y", ["sent"] + CONTROLS, ["year"]), "sent")})
    sec.append({"結果": "D2，不設討論量門檻",
                **one_sided(lm.fe_ols(d2.dropna(subset=["sent"]), "oc", ["sent"] + CONTROLS, ["date"]), "sent")})
    s = w.dropna(subset=["sent", "ar1_z"])
    sec.append({"結果": "corr(sent, ar(t−1)/σ)", "coef": float(np.corrcoef(s["sent"], s["ar1_z"])[0, 1]), "n": len(s)})
    # 照 sent 的正負號做多或放空，開盤進場、收盤出場，扣現股當沖成本
    c = cfg["costs"]
    cost = 2 * c["fee_rate"] * c["fee_discount"] + c["daytrade_tax_rate"]
    tr = d2[(d2["ptt_abn_pre_early"] >= D.D2_ABN_MIN)].dropna(subset=["sent", "oc"])
    tr = tr[tr["sent"] != 0]
    gross = np.sign(tr["sent"]) * np.expm1(tr["oc"])
    sec.append({"結果": "D2 照正負號當沖：每筆平均毛報酬", "coef": float(gross.mean()),
                "t": float(gross.mean() / gross.std() * np.sqrt(len(gross))), "n": len(gross)})
    sec.append({"結果": "D2 照正負號當沖：扣成本後", "coef": float(gross.mean() - cost), "n": len(gross)})
    sec = pd.DataFrame(sec)
    sec.to_csv(output_path(cfg, "large_moves", "direction_secondary.csv"), index=False, encoding="utf-8-sig")
    print("\n次要結果：\n" + sec.round(4).to_string(index=False))
    print(f"\n樣本：D1 事件 {int(ev['sent'].notna().sum())} / {len(ev)}；D2 {int(f2.nobs)} 天")


def cmd_gapcheck(cfg):
    """計畫「事後檢查」一節：D1、D2 加上開盤跳空。"""
    if not committed(REGISTERED):
        sys.exit("計畫或程式有沒 commit 的修改；先 commit 再跑")
    w = build(cfg)
    ev = w[w["main"]].assign(y=lambda x: x["up"].astype(float)).dropna(subset=["sent", "gap_z", "oc"])
    d2 = w[w["eligible"] & ~w["followon"] & ~w["open_locked"] & (w["ptt_abn_pre_early"] >= D.D2_ABN_MIN)]
    d2 = d2.dropna(subset=["sent", "gap_z"])
    rows = [
        {"檢查": "D2（同一樣本，不加跳空）", **one_sided(lm.fe_ols(d2, "oc", ["sent"] + CONTROLS, ["date"]), "sent")},
        {"檢查": "C1：D2 加跳空", **one_sided(lm.fe_ols(d2, "oc", ["sent", "gap_z"] + CONTROLS, ["date"]), "sent")},
        {"檢查": "D1（同一樣本，不加跳空）", **one_sided(lm.fe_ols(ev, "y", ["sent"] + CONTROLS, ["year"]), "sent")},
        {"檢查": "C2：D1 加跳空", **one_sided(lm.fe_ols(ev, "y", ["sent", "gap_z"] + CONTROLS, ["year"]), "sent")},
        {"檢查": "C3：D1 樣本，情緒 → 跳空 gap_z", **one_sided(lm.fe_ols(ev, "gap_z", ["sent"] + CONTROLS, ["year"]), "sent")},
        {"檢查": "C3：D1 樣本，情緒 → 盤中 oc", **one_sided(lm.fe_ols(ev, "oc", ["sent"] + CONTROLS, ["year"]), "sent")},
    ]
    f = lm.fe_ols(d2, "oc", ["sent", "gap_z"] + CONTROLS, ["date"])
    out = pd.DataFrame(rows)
    out.to_csv(output_path(cfg, "large_moves", "direction_gapcheck.csv"), index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))
    print(f"\nC1 的跳空係數 {f.params['gap_z']:.5f}（t = {f.tvalues['gap_z']:.2f}）；"
          f"D2 樣本中 corr(sent, gap_z) = {np.corrcoef(d2['sent'], d2['gap_z'])[0, 1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["texts", "score", "analyze", "gapcheck"])
    a = ap.parse_args()
    cfg = load_config()
    {"texts": cmd_texts, "score": cmd_score, "analyze": cmd_analyze, "gapcheck": cmd_gapcheck}[a.cmd](cfg)


if __name__ == "__main__":
    main()

"""大漲跌日 P1 的預先登記檢定（見 docs/prereg/PREREGISTRATION_LARGE_MOVES.md）。設定全部寫死。

    python scripts/17_large_moves_prereg.py power          # 檢定力：只用開發期（2019–2023）
    python scripts/17_large_moves_prereg.py check A        # 只檢查資料涵蓋（每月的合格股票日、則數），不跑回歸
    python scripts/17_large_moves_prereg.py final A        # L1、L2：事件日 2024-01-01 .. 2026-09-24，只跑一次
    python scripts/17_large_moves_prereg.py final B        # 同上：事件日 2026-10-01 .. 2027-09-30，期間結束、資料補齊後只跑一次

L1（主要）：事件(t) ~ 開盤前 PTT 異常（前一天收盤到 08:30）+ 控制，開盤前 PTT 的係數 > 0，單尾 α = 0.05。
L2（次要）：開發期 P1 的原設定（開盤前到 09:00，沒有重大訊息）。
事件定義、名稱比對、時間窗、固定效果與雙向 cluster 都沿用 src/pttsent/large_moves.py（開發期已 commit 的版本）。
"""
import argparse
import importlib.util
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
from pttsent import large_moves as lm  # noqa: E402
from pttsent import mops  # noqa: E402
from pttsent import volume as v  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402

_spec = importlib.util.spec_from_file_location("s16", ROOT / "scripts" / "16_large_moves.py")
s16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s16)

PERIODS = {"A": ("2024-01-01", "2026-09-24"), "B": ("2026-10-01", "2027-09-30")}
MOPS_DIR = {"A": "t05st01", "B": "t05st01_2027", "dev": "t05st01"}   # data/mops/ 底下的資料夾
DEV = (lm.EVENT_START, lm.EVENT_END)
ALPHA = 0.05
L1_X = ["ptt_abn_pre_early", "ptt_abn_pre_late", "ptt_abn_pre5", "news_log_preopen", "news_log_pre5",
        "mops_log_preopen", "mops_log_pre5", "abs_z1", "log_sigma", "lock1", "lvol_abn1"]
L2_X = lm.P12_X
REGISTERED = ["docs/prereg/PREREGISTRATION_LARGE_MOVES.md", "scripts/17_large_moves_prereg.py",
              "scripts/16_large_moves.py", "src/pttsent/large_moves.py", "src/pttsent/mops.py",
              "docs/large_moves/names_rules.csv"]


def committed(paths) -> bool:
    if not all((ROOT / f).exists() for f in paths):
        return False
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def build(cfg, first: str, last: str, mops_dir: str) -> pd.DataFrame:
    """事件日在 [first, last] 的合格股票日，加上 L1、L2 用到的變數。只用到 t 開盤前的資料（加上 t 當天的事件旗標）。"""
    u = lm.universe(cfg["data_dir"])
    rules = s16.load_rules()
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                            columns=["date", "code", "market", "close", "volume", "base", "ret", "mkt_ret",
                                     "limit_up_close", "limit_down_close"])
    days = pd.DatetimeIndex(sorted(panel.loc[panel["market"] == "TWSE", "date"].unique()))
    panel = panel[panel["code"].isin(set(u["code"])) & (panel["date"] <= last)]
    d = lm.stock_days(panel, days, v.load_disposition(cfg["data_dir"]), u["code"])
    d["locked"] = d["limit_up_close"] | d["limit_down_close"]
    d["dt_intraday"] = np.nan
    d = lm.stock_lags(d)
    d = d[(d["date"] >= first) & (d["date"] <= last) & d["eligible"] & ~d["followon"]].copy()

    # 文字資料只讀到 t 當天 09:00 以前：最後一個事件日的開盤
    end = str(pd.Timestamp(last) + pd.Timedelta(hours=9))
    start = str(days[max(days.searchsorted(pd.Timestamp(first)) - lm.BASE[0] - 2, 0)].date())
    years = range(int(start[:4]), int(last[:4]) + 1)
    ptt = lm.ptt_items(work_path(cfg, "interim", "ptt", "x").parent, years, rules,
                       cfg["sentiment"]["max_comment_lag_days"], end)
    ptt = ptt[ptt["time"] >= start]
    news = s16.load_news_cached(cfg)
    news = lm.news_items(news[(news["time"] >= start) & (news["time"] < end)], rules)
    ann = mops.load(work_path(cfg, "mops", mops_dir, "x").parent)
    ann = ann[(ann["time"] >= start) & (ann["time"] < end)][["code", "time"]]

    d = lm.add_windows(d, ptt, days, "ptt", start=start)
    d = lm.add_windows(d, news, days, "news", start=start)
    d = lm.add_windows(d, ann, days, "mops", start=start)
    d = lm.outcomes(d)
    for w in ("preopen", "pre5"):
        d[f"mops_log_{w}"] = np.log1p(d[f"mops_{w}"])
    split = lm.preopen_split(d, ptt, days)
    d["ptt_abn_pre_early"] = np.log1p(split["pre_early"]) - d["ptt_base"]
    d["ptt_abn_pre_late"] = np.log1p(split["pre_late"]) - d["ptt_base"]
    d["code_year"] = d["code"] + "_" + d["date"].dt.year.astype(str)
    d.attrs["n_ptt"], d.attrs["n_news"], d.attrs["n_mops"] = len(ptt), len(news), len(ann)
    return d


def fit(d: pd.DataFrame, xs, x: str) -> dict:
    f = lm.p12(d, xs)
    sd = d.loc[d["eligible"] & ~d["followon"], x].std()
    return {"coef": f.params[x], "se": f.bse[x], "t": f.tvalues[x], "p": 1 - norm.cdf(f.tvalues[x]),
            "n": int(f.nobs), "events": int(d["main"].sum()), "base_rate": float(d["main"].mean()),
            "effect_1sd_pp": 100 * f.params[x] * sd}


def cmd_power(cfg, period):
    """開發期的 L1 t 值，依交易日數換算評估期的預期 t 與檢定力（解析式，單尾 α = 0.05）。"""
    d = build(cfg, *DEV, MOPS_DIR["dev"])
    dev = fit(d, L1_X, "ptt_abn_pre_early")
    days = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "market"])
    days = pd.DatetimeIndex(sorted(days.loc[days["market"] == "TWSE", "date"].unique()))
    n_dev = int(((days >= DEV[0]) & (days <= DEV[1])).sum())
    z = norm.ppf(1 - ALPHA)
    print(f"開發期 L1：係數 {dev['coef']:.4f}，t = {dev['t']:.2f}，{n_dev} 個交易日")
    for k, (a, b) in PERIODS.items():
        n = int(((days >= a) & (days <= b)).sum()) if k == "A" else 245   # B 還沒有交易日曆，以 245 天估
        for share in (1.0, 0.5):
            t = dev["t"] * share * np.sqrt(n / n_dev)
            print(f"{k}（{n} 天）效果 × {share}：預期 t = {t:.2f}，檢定力 {norm.cdf(t - z):.0%}")


def cmd_check(cfg, period):
    a, b = PERIODS[period]
    d = build(cfg, a, b, MOPS_DIR[period])
    print(f"合格股票日 {len(d):,}、主要事件 {int(d['main'].sum())}；PTT {d.attrs['n_ptt']:,}、"
          f"鉅亨 {d.attrs['n_news']:,}、重大訊息 {d.attrs['n_mops']:,} 則")
    m = d.groupby(d["date"].dt.to_period("M")).agg(stock_days=("code", "size"), ptt_missing=("ptt_base", lambda s: s.isna().mean()))
    print(m.to_string())
    for x in L1_X:
        print(f"{x}: 缺值 {d[x].isna().mean():.4f}")


def cmd_final(cfg, period):
    a, b = PERIODS[period]
    out = output_path(cfg, "large_moves", f"prereg_{period}.json")
    if not committed(REGISTERED):
        sys.exit("要在預先登記 commit 之後才能跑，這些檔案不存在或有未 commit 的改動：" + "、".join(REGISTERED))
    if date.today() <= date.fromisoformat(b):
        sys.exit(f"評估期到 {b} 才結束，現在不能跑")
    if out.exists():
        sys.exit(f"{out} 已經存在：這個檢定已經跑過，不能再跑")
    d = build(cfg, a, b, MOPS_DIR[period])
    res = {"period": period, "window": [a, b],
           "L1": fit(d, L1_X, "ptt_abn_pre_early"), "L2": fit(d, L2_X, "ptt_abn_preopen")}
    for k in ("L1", "L2"):
        res[k]["pass"] = bool(res[k]["p"] < ALPHA)
    res["run_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["power", "check", "final"])
    ap.add_argument("period", nargs="?", choices=list(PERIODS))
    a = ap.parse_args()
    if a.cmd != "power" and not a.period:
        ap.error("check / final 要指定 A 或 B")
    cfg = load_config()
    {"power": cmd_power, "check": cmd_check, "final": cmd_final}[a.cmd](cfg, a.period)


if __name__ == "__main__":
    main()

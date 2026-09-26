"""[標的] 事件研究，第 2 步：開發期（2016–2023）的分析。最終測試期（2024 以後）的事件一律不算報酬。

    python scripts/08_target_dev.py

主要（之後要預先登記的）：
  H1 發文後 5 天、H2 發文後 20 天的市場調整累積報酬，看多文 − 看空文 > 0（普通股，雙向叢集標準誤）
描述：H3 發文前 20 天的累積報酬（作者有沒有追漲殺跌）。
穩健性：去掉極端值、市場模型、對照組合、原始報酬、進場收盤漲跌停買不到的排除、日曆時間組合、1／60 天、各年。
最後用開發期的變異估計最終測試期能偵測到多大的差距。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import event_study as es  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
HORIZONS = (1, 5, 20, 60)


def fmt(r: dict) -> str:
    return (f"看多 {r['mean_bull']:+.2%}（{r['n_bull']}）  看空 {r['mean_bear']:+.2%}（{r['n_bear']}）  "
            f"差 {r['spread']:+.2%} ± {r['se']:.2%}  t = {r['t']:+.2f}  單尾 p = {r['p_one_sided']:.3f}  雙尾 p = {r['p_two_sided']:.3f}")


def main():
    cfg = load_config()
    ev = pd.read_parquet(work_path(cfg, "processed", "target_events.parquet"))
    n_final = int(((ev["split"] == "final") & ev["common"]).sum())
    n_final_bear = int(((ev["split"] == "final") & ev["common"] & (ev["direction"] == "bearish")).sum())
    ev = ev[(ev["split"] == "dev") & ev["common"]].reset_index(drop=True)   # 最終測試期在這裡就排除
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"))
    market = pd.read_parquet(work_path(cfg, "prices", "market.parquet"))
    days = pd.DatetimeIndex(sorted(panel.loc[panel["market"] == "TWSE", "date"].unique()))
    common = panel.loc[panel["common"], "code"].unique()
    w = es.wide(panel[panel["common"]], market, days, common)
    ar = es.abnormal_returns(ev, w, HORIZONS)
    lim = panel.set_index(["code", "date"])[["limit_up_close", "limit_down_close"]]
    ar = ar.join(lim, on=["code", "entry_date"])
    out_dir = output_path(cfg, "target", "dev", "x").parent
    ar.to_parquet(out_dir / "events_with_returns.parquet", index=False)
    print(f"開發期事件 {len(ar):,}（看多 {(ar.direction == 'bullish').sum():,}、看空 {(ar.direction == 'bearish').sum():,}），"
          f"{ar['entry_date'].min().date()} .. {ar['entry_date'].max().date()}，{ar['code'].nunique()} 檔股票\n")

    results = {}
    print("== 主要：市場調整累積報酬，看多 − 看空")
    for h in (5, 20):
        results[f"H{1 if h == 5 else 2}_car_{h}"] = r = es.spread_test(ar, f"car_{h}")
        print(f"{h:>2} 天  {fmt(r)}")

    print("\n== 描述：發文前與發文當下（雙尾）")
    for col, label in [("pre_20", "發文前 20 天"), ("pre_5", "發文前 5 天"), ("day0", "發文當下（entry 當天）")]:
        results[col] = r = es.spread_test(ar, col)
        print(f"{label:14s}{fmt(r)}")

    print("\n== 穩健性")
    locked = ((ar["direction"] == "bullish") & ar["limit_up_close"].astype("boolean").fillna(False)) \
        | ((ar["direction"] == "bearish") & ar["limit_down_close"].astype("boolean").fillna(False))
    tradable = ~locked.astype(bool)
    checks = []
    for h in HORIZONS:
        checks += [(f"car_{h}", f"市場調整 {h} 天", ar, None)]
    for h in (5, 20):
        checks += [(f"car_{h}", f"市場調整 {h} 天，去掉 1% 極端值", ar, 0.01),
                   (f"mm_{h}", f"市場模型 {h} 天", ar, None),
                   (f"bm_{h}", f"對照組合 {h} 天", ar, None),
                   (f"raw_{h}", f"原始報酬 {h} 天", ar, None),
                   (f"car_{h}", f"市場調整 {h} 天，排除進場收盤漲跌停", ar[tradable], None)]
    rob = []
    for col, label, d, win in checks:
        r = es.spread_test(d, col, winsor=win)
        rob.append({"檢查": label, **{k: r[k] for k in ("n_bull", "n_bear", "mean_bull", "mean_bear", "spread", "t", "p_two_sided")}})
    rob = pd.DataFrame(rob)
    print(rob.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    rob.to_csv(out_dir / "robustness.csv", index=False)

    print("\n== 進場收盤就漲停（看多）／跌停（看空）的事件：漲跌停造成的延續，而且買不到")
    for side in ("bullish", "bearish"):
        m = ar["direction"] == side
        for h in (1, 5, 20):
            a_, b_ = ar.loc[m & ~tradable, f"car_{h}"], ar.loc[m & tradable, f"car_{h}"]
            print(f"  {'看多' if side == 'bullish' else '看空'} {h:>2} 天：鎖住的 {a_.mean():+.2%}（{a_.notna().sum()}）  其他 {b_.mean():+.2%}（{b_.notna().sum()}）")
    results["share_locked"] = float((~tradable).mean())

    print("\n== 日曆時間組合（每天：持有中的看多事件 − 看空事件，Newey-West）")
    for h in (5, 20):
        s = es.calendar_time(ar, w, h)
        results[f"calendar_{h}"] = r = es.newey_west_mean(s, lags=h)
        print(f"{h:>2} 天  每日平均 {r['mean_daily']:+.4%}（約 {h} 天 {r['mean_daily'] * h:+.2%}）  t = {r['t']:+.2f}  "
              f"p = {r['p_two_sided']:.3f}  {r['n_days']} 天")

    print("\n== 各年（市場調整，看多 − 看空）")
    ar["year"] = ar["time"].dt.year      # 發文年份（年底的文 entry 可能落在隔年初）
    by_year = pd.DataFrame({y: {**{f"spread_{h}": es.spread_test(g, f"car_{h}")["spread"] for h in (5, 20)},
                                **{f"tradable_{h}": es.spread_test(g[tradable[g.index]], f"car_{h}")["spread"] for h in (5, 20)},
                                "n": len(g), "n_bear": int((g.direction == "bearish").sum()),
                                "pre_20": es.spread_test(g, "pre_20")["spread"]}
                            for y, g in ar.groupby("year")}).T
    print(by_year.round(4).to_string())
    by_year.to_csv(out_dir / "by_year.csv")

    print("\n== 能不能賺錢：只買看多文，扣來回成本")
    cost = cfg["costs"]["fee_rate"] * cfg["costs"]["fee_discount"] * 2 + cfg["costs"]["tax_rate"]
    for h in (5, 20):
        b = ar.loc[(ar.direction == "bullish") & tradable, f"car_{h}"].dropna()
        print(f"{h:>2} 天：市場調整平均 {b.mean():+.2%}，扣掉來回成本 {cost:.3%} 後 {b.mean() - cost:+.2%}（{len(b)} 筆）")

    print("\n== 最終測試期（2024/01–2025/01）的檢定力估計")
    power = {}
    for h in (5, 20):
        r = results[f"H{1 if h == 5 else 2}_car_{h}"]
        # 叢集後的標準誤依樣本數比例放大：SE_final ≈ SE_dev × sqrt(有效樣本比)
        eff_dev = 1 / (1 / r["n_bull"] + 1 / r["n_bear"])
        eff_fin = 1 / (1 / max(n_final - n_final_bear, 1) + 1 / max(n_final_bear, 1))
        se_fin = r["se"] * np.sqrt(eff_dev / eff_fin)
        mde = (norm.ppf(1 - 0.025) + norm.ppf(0.8)) * se_fin     # Holm 第一步 α = 0.025（單尾），80% 檢定力
        power[h] = {"se_final": se_fin, "mde": mde}
        print(f"{h:>2} 天：最終測試約 {n_final} 筆（看空 {n_final_bear}），標準誤約 {se_fin:.2%}，"
              f"能偵測到的最小差距約 {mde:.2%}（開發期實際差距 {r['spread']:+.2%}）")

    json.dump({"results": results, "power": power, "n_final": n_final, "n_final_bear": n_final_bear},
              open(out_dir / "summary.json", "w"), indent=2, default=float)
    print(f"\n結果 -> {out_dir}")


if __name__ == "__main__":
    main()

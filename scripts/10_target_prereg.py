"""[標的] 事件研究的預先登記檢定（見 PREREGISTRATION_TARGET.md）。

    python scripts/10_target_prereg.py            # 開發期（2016–2023），確認和 08_target_dev.py 一致
    python scripts/10_target_prereg.py --final    # 最終測試期（2024-01-01 以後）：預先登記 commit 後只跑一次

H1：car_5、H2：car_20，看多 − 看空 > 0（雙向叢集、單尾），Holm 校正整體 α = 0.05。
H3：pre_20 看多 − 看空 ≠ 0（雙尾）。
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import event_study as es  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402


def holm(p: dict, alpha: float = 0.05) -> dict:
    """Holm 逐步校正：p 值由小到大和 alpha/(m-i) 比，一旦沒過，後面都不過。"""
    order = sorted(p, key=p.get)
    out, ok = {}, True
    for i, k in enumerate(order):
        ok = ok and p[k] < alpha / (len(p) - i)
        out[k] = ok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    split = "final" if a.final else "dev"
    ev = pd.read_parquet(work_path(cfg, "processed", "target_events.parquet"))
    ev = ev[(ev["split"] == split) & ev["common"]].reset_index(drop=True)
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"))
    market = pd.read_parquet(work_path(cfg, "prices", "market.parquet"))
    days = pd.DatetimeIndex(sorted(panel.loc[panel["market"] == "TWSE", "date"].unique()))
    w = es.wide(panel[panel["common"]], market, days, panel.loc[panel["common"], "code"].unique())
    ar = es.abnormal_returns(ev, w, horizons=(5, 20), pre=(20,))
    lim = panel.set_index(["code", "date"])[["limit_up_close"]]
    ar = ar.join(lim, on=["code", "entry_date"])
    print(f"{split}：{len(ar):,} 個事件（看多 {(ar.direction == 'bullish').sum():,}、看空 {(ar.direction == 'bearish').sum():,}），"
          f"entry {ar['entry_date'].min().date()} .. {ar['entry_date'].max().date()}\n")

    r = {"H1": es.spread_test(ar, "car_5"), "H2": es.spread_test(ar, "car_20"), "H3": es.spread_test(ar, "pre_20")}
    passed = holm({k: r[k]["p_one_sided"] for k in ("H1", "H2")})
    for k, h in (("H1", 5), ("H2", 20)):
        x = r[k]
        print(f"{k}（{h} 天）：看多 {x['mean_bull']:+.2%}（{x['n_bull']}）  看空 {x['mean_bear']:+.2%}（{x['n_bear']}）  "
              f"差 {x['spread']:+.2%}  t = {x['t']:+.2f}  單尾 p = {x['p_one_sided']:.4f}  "
              f"-> {'通過' if passed[k] else '未通過'}（Holm）")
    x = r["H3"]
    print(f"H3（發文前 20 天）：差 {x['spread']:+.2%}  t = {x['t']:+.2f}  雙尾 p = {x['p_two_sided']:.4f}  "
          f"-> {'有差別' if x['p_two_sided'] < 0.05 else '沒有顯著差別'}")

    cost = cfg["costs"]["fee_rate"] * cfg["costs"]["fee_discount"] * 2 + cfg["costs"]["tax_rate"]
    trade = {}
    for h in (5, 20):
        b = ar.loc[(ar.direction == "bullish") & ~ar["limit_up_close"].fillna(False), f"car_{h}"].dropna()
        trade[h] = {"n": len(b), "mean": b.mean(), "net_of_cost": b.mean() - cost}
        print(f"只買看多文 {h} 天（排除收盤漲停）：市場調整 {b.mean():+.2%}，扣成本 {cost:.3%} 後 {b.mean() - cost:+.2%}")

    out = output_path(cfg, "target", f"prereg_{split}", "result.json")
    json.dump({"results": r, "holm_passed": passed, "trade": trade}, open(out, "w"), indent=2, default=float)
    ar.to_parquet(out.with_name("events_with_returns.parquet"), index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

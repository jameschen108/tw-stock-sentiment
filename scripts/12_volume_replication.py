"""開盤前 PTT 討論量 → 當天成交量／振幅：2025–2026 的重複驗證（見 PREREGISTRATION_VOLUME_REPLICATION.md）。

    python scripts/12_volume_replication.py            # 只檢查資料涵蓋（每檔可用天數），不算任何檢定
    python scripts/12_volume_replication.py --final    # 重複驗證：預先登記 commit 後只跑一次

完全沿用已登記的 11_volume_prereg.py 的特徵、模型與檢定（直接載入它的 build 與 evaluate），
只換兩樣：評估期間（目標日 2025-01-02 .. 2026-09-24）與美股資料（data/prices/us_2026/，期間較長）。
訓練資料是 2019-02 起到每個區塊開始前的所有列（含 2024）。
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("volume_prereg", ROOT / "scripts" / "11_volume_prereg.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)

REPLICATION = ("2025-01-01", "2026-09-24")
US_DIR = "us_2026"
REGISTERED = ["PREREGISTRATION_VOLUME_REPLICATION.md", "scripts/12_volume_replication.py"] + p.REGISTERED
_work_path = p.work_path


def work_path(cfg, *parts):
    """11 的 build 固定讀 data/prices/us/；重複驗證改讀期間較長的 us_2026/，其餘路徑不變。"""
    if parts[:2] == ("prices", "us"):
        parts = ("prices", US_DIR) + parts[2:]
    return _work_path(cfg, *parts)


p.work_path = work_path
p.FINAL = REPLICATION        # evaluate(final=True) 用這個期間


def coverage(tables) -> None:
    for tk in p.TICKERS:
        for y in p.TARGETS:
            d = tables[tk][[y] + p.v.baseline_cols(y) + ["pre_abn"]].dropna()
            ev = d.loc[REPLICATION[0]:REPLICATION[1]]
            print(f"{tk} {p.TARGETS[y]}：評估期可用 {len(ev)} 天（{ev.index.min().date()} .. {ev.index.max().date()}），"
                  f"之前可訓練 {len(d[d.index < REPLICATION[0]])} 天")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.final and not p.committed(REGISTERED):
        sys.exit("重複驗證要在預先登記 commit 之後才能跑，這些檔案有未 commit 的改動：" + "、".join(REGISTERED))
    cfg = p.load_config()
    tables, days = p.build(cfg)
    n_days = int(((days >= REPLICATION[0]) & (days <= REPLICATION[1])).sum())
    print(f"重複驗證期間 {REPLICATION[0]} .. {REPLICATION[1]}：{n_days} 個交易日\n")
    if not a.final:
        coverage(tables)
        return

    res = {(y, tk): p.evaluate(tables[tk], y, True) for y in p.TARGETS for tk in p.TICKERS}
    fs = {k: r.pop("f") for k, r in res.items()}
    fs_gap = {k: r.pop("f_gap") for k, r in res.items()}

    def pool(f, y):
        return pd.concat([f[(y, tk)] for tk in p.TICKERS], axis=1, join="inner").mean(axis=1)

    h1, h3 = p.v.mean_test(pool(fs, "y_lv"), p.NW_LAGS), p.v.mean_test(pool(fs, "y_vol"), p.NW_LAGS)
    h1["pass"], h3["pass"] = h1["p_one_sided"] < 0.05, h3["p_one_sided"] < 0.05
    h2 = p.holm({tk: res[("y_lv", tk)]["p_one_sided"] for tk in p.TICKERS})
    gap = p.v.mean_test(pool(fs_gap, "y_lv"), p.NW_LAGS)

    print(f"H1（主要）三檔平均 log 成交量：CW t = {h1['t']:+.2f}，單尾 p = {h1['p_one_sided']:.4f} "
          f"-> {'通過' if h1['pass'] else '未通過'}（三檔共同 {h1['n']} 天）")
    for tk in p.TICKERS:
        r = res[("y_lv", tk)]
        print(f"H2 {tk} log 成交量：CW t = {r['t']:+.2f}，單尾 p = {r['p_one_sided']:.4f} "
              f"-> {'通過' if h2[tk] else '未通過'}（Holm）")
    print(f"H3 三檔平均 log 振幅：CW t = {h3['t']:+.2f}，單尾 p = {h3['p_one_sided']:.4f} "
          f"-> {'通過' if h3['pass'] else '未通過'}")
    print(f"解讀（非假設）三檔平均 log 成交量，再加開盤跳空：CW t = {gap['t']:+.2f}，單尾 p = {gap['p_one_sided']:.4f}\n")

    rows = [{"股票": tk, "目標": p.TARGETS[y], "天數": r["n"], "MSE 變化 %": 100 * r["mse_change"],
             "CW t": r["t"], "單尾 p": r["p_one_sided"], "係數": r["coef"], "係數 t": r["coef_t"],
             "1 SD 效果 %": 100 * r["effect_1sd"], "加跳空後 MSE 變化 %": 100 * r["mse_change_with_gap"],
             "加跳空後 CW t": r["cw_t_with_gap"]} for (y, tk), r in res.items()]
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    out = p.output_path(cfg, "volume", "replication", "result.json")
    json.dump({"period": REPLICATION, "H1": h1, "H2_holm": h2, "H3": h3, "with_gap_pooled_volume": gap,
               "by_stock": {f"{y}_{tk}": r for (y, tk), r in res.items()}},
              open(out, "w"), ensure_ascii=False, indent=2, default=float)
    pd.DataFrame({f"{y}_{tk}": f for (y, tk), f in fs.items()}).to_parquet(out.with_name("cw_loss.parquet"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

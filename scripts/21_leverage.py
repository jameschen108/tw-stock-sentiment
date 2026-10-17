"""研究五：槓桿 ETF 對 TAIEX 報酬指數（見 docs/prereg/PREREGISTRATION_LEVERAGE.md）。

    python scripts/21_leverage.py dev        # 開發期 2003–2023
    python scripts/21_leverage.py power      # 檢定力：只用開發期的 t 值
    python scripts/21_leverage.py check A    # 只檢查資料涵蓋（交易日、缺值、異常日的日期），不算報酬
    python scripts/21_leverage.py final A    # 2024-01-01 .. 2026-09-24，只跑一次（2026-10-17 已跑過）

評估期 B（2026-10 .. 2029-09）在 2026-10-17 由使用者決定取消，見登記文件的修改紀錄。

主要假設 H1：C2 的每日對數超額報酬（策略 − TAIEX TR）平均 > 0，Newey-West t（20 期），單尾 α = 0.05。
次要 H2：C1、C3 同樣的檢定，兩個一起做 Holm 校正。

候選策略與成本設定在 src/pttsent/leverage.py，看結果之前寫死；這裡把它和依賴檔的 sha256 寫進
output/leverage/dev_meta.json。輸出：dev_summary.csv（全期與兩個子期間）、dev_stress.csv（壓力期間）。
1999–2002 只有價格指數，1x 和大盤都少了股息，只當描述，不進判定。
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pttsent import leverage as lv  # noqa: E402
from pttsent.backtest import etf_costs, weights_backtest  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402

DEPS = ["src/pttsent/leverage.py", "src/pttsent/etf.py", "src/pttsent/backtest.py", "docs/leverage/splits.csv"]
PERIODS = {"2003–2023": ("2003-01-02", "2023-12-31"),
           "2003–2014 合成": ("2003-01-02", "2014-12-31"),
           "2015–2023 實際 ETF": ("2015-01-06", "2023-12-31")}
FINAL = {"A": ("2024-01-01", "2026-09-24")}
REGISTERED = ["docs/prereg/PREREGISTRATION_LEVERAGE.md", "scripts/21_leverage.py", *DEPS]
COSTS = {"fee_rate": 0.001425, "fee_discount": 1.0, "etf_tax_rate": 0.001}   # 登記時的 config.yaml
ALPHA, NW_LAGS = 0.05, 20
DEV_T = {"C1": 1.896, "C2": 1.404, "C3": 0.967}   # 開發期 2003–2023 的 NW t（dev_summary.csv）
DEV_YEARS = 21.0
STRESS = {"2008": ("2008-01-01", "2008-12-31"), "2015-04..08": ("2015-04-28", "2015-08-24"),
          "2018-10": ("2018-10-01", "2018-10-31"), "2020-02..03": ("2020-02-20", "2020-03-19"),
          "2022": ("2022-01-01", "2022-12-31")}


def run(inp, cfg, start, end, bench="bench") -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """回傳 {策略: 每日 net（簡單淨報酬）、cost、trade、exposure，索引 = 報酬發生的那天}, 大盤對數報酬。

    t 收盤的交易成本和權重記在 t+1 那列，跟它影響的報酬放在一起。
    """
    ret, tradable = lv.assets(inp, start, end, bench)
    days = ret.index
    buy, sell = etf_costs(cfg)
    out, bands = {}, lv.bands()
    for name, w in lv.strategies(inp["signal"], days).items():
        bt = weights_backtest(ret.shift(-1), w, buy, sell, band=bands[name], tradable=tradable)
        bt = bt.assign(exposure=bt["w_x1"] + 2 * bt["w_x2"])[["net", "cost", "trade", "exposure"]].iloc[:-1]
        bt.index = days[1:]
        out[name] = bt
    return out, inp[bench].reindex(days[1:])


def summarize(nets, bench, start, end, boot=True) -> pd.DataFrame:
    rows = {}
    for name, bt in nets.items():
        b = bt.loc[start:end]
        r = lv.evaluate(b["net"], bench)
        if boot:
            r["p_win_3y_boot"] = lv.bootstrap_win(np.log1p(b["net"]) - bench.reindex(b.index))
        r.update(avg_exposure=b["exposure"].mean(), trades=int(b["trade"].sum()), cost=b["cost"].sum())
        rows[name] = r
    return pd.DataFrame(rows).T


def cmd_dev(cfg):
    out_dir = output_path(cfg, "leverage", "x").parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = {"run_at": datetime.now().astimezone().isoformat(), "git_head": head,
            "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in DEPS}}
    (out_dir / "dev_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "code", "close", "base"])
    inp = lv.load_inputs(cfg, panel)
    nets, bench = run(inp, cfg, "2003-01-02", "2023-12-31")
    tables = []
    for label, (s, e) in PERIODS.items():
        t = summarize(nets, bench, s, e, boot=(label == "2003–2023"))
        t.insert(0, "period", label)
        tables.append(t)
    summary = pd.concat(tables)
    summary.to_csv(out_dir / "dev_summary.csv")

    stress = {}
    for label, (s, e) in STRESS.items():
        row = {name: np.expm1(np.log1p(bt["net"].loc[s:e]).sum()) for name, bt in nets.items()}
        row["TAIEX TR"] = np.expm1(bench.loc[s:e].sum())
        stress[label] = row
    stress = pd.DataFrame(stress).T
    stress.to_csv(out_dir / "dev_stress.csv")

    early, bench_p = run(inp, cfg, "1999-01-05", "2002-12-31", bench="price")
    desc = summarize(early, bench_p, "1999-01-05", "2002-12-31", boot=False)
    desc.to_csv(out_dir / "dev_1999_2002_price_only.csv")

    pd.set_option("display.width", 200)
    cols = ["period", "cagr", "bench_cagr", "excess_cagr", "ann_vol", "max_dd", "worst_12m", "win_3y",
            "nw_t", "p_win_3y_boot", "avg_exposure", "trades", "cost"]
    print(summary[cols].round(3).to_string())
    print("\n壓力期間累積報酬")
    print(stress.round(3).to_string())
    print("\n1999–2002（只有價格指數，只當描述）")
    print(desc[cols[1:9]].round(3).to_string())
    print(f"\n輸出在 {out_dir}；依賴檔 sha256 在 dev_meta.json")


def committed(paths) -> bool:
    if not all((ROOT / f).exists() for f in paths):
        return False
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def cmd_power(cfg, period):
    """t 大致和年數的平方根成正比：把開發期 21 年的 t 換算成評估期的預期 t。"""
    z = norm.ppf(1 - ALPHA)
    for k, (a, b) in FINAL.items():
        years = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        for name, t0 in DEV_T.items():
            for share in (1.0, 0.5):
                t = t0 * share * np.sqrt(years / DEV_YEARS)
                print(f"{k}（{years:.1f} 年）{name} 效果 × {share}：預期 t = {t:.2f}，檢定力 {norm.cdf(t - z):.0%}")


def _period_inputs(cfg, period):
    first, last = FINAL[period]
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "code", "close", "base"])
    inp = lv.load_inputs(cfg, panel)
    prev = inp["bench"].index[inp["bench"].index < first][-1]   # 前一個交易日收盤建倉，評估期每一天都有報酬
    return inp, prev, first, last


def cmd_check(cfg, period):
    inp, prev, first, last = _period_inputs(cfg, period)
    days = inp["bench"].loc[first:last].index
    e = inp["etfs"].reindex(days)
    print(f"評估期 {first} .. {last}：TAIEX TR 有 {len(days)} 個交易日，最後一天 {days[-1].date()}")
    print(f"建倉日（前一個交易日）：{prev.date()}")
    for c in e.columns:
        miss = e.index[e[c].isna()]
        print(f"{c}：缺值 {len(miss)} 天 {[d.date().isoformat() for d in miss]}")
    for c, k in (("0050", 1), ("00631L", 2)):
        bad = lv.etf.check_vs_index(inp["etfs"][c].loc[first:last], inp["bench"], k)
        print(f"{c} 和 {k} 倍 TR 差超過 3% 的日子（只列日期）：{[d.date().isoformat() for d in bad.index]}")
    print("costs:", {k: cfg["costs"][k] for k in COSTS}, "登記：", COSTS)


def cmd_final(cfg, period):
    out = output_path(cfg, "leverage", f"prereg_{period}.json")
    if not committed(REGISTERED):
        sys.exit("要在預先登記 commit 之後才能跑，這些檔案不存在或有未 commit 的改動：" + "、".join(REGISTERED))
    if {k: cfg["costs"][k] for k in COSTS} != COSTS:
        sys.exit(f"config.yaml 的成本和登記時不同：登記 {COSTS}")
    first, last = FINAL[period]
    if date.today() <= date.fromisoformat(last):
        sys.exit(f"評估期到 {last} 才結束，現在不能跑")
    if out.exists():
        sys.exit(f"{out} 已經存在：這個檢定已經跑過，不能再跑")
    inp, prev, first, last = _period_inputs(cfg, period)
    if inp["bench"].index[-1] < pd.Timestamp(last) - pd.Timedelta(days=7) or inp["etfs"].index[-1] < inp["bench"].loc[:last].index[-1]:
        sys.exit("資料還沒補到評估期結束：先補報酬指數與股價面板")
    nets, bench = run(inp, cfg, prev, last)
    res = {"period": period, "window": [first, last], "entry_day": prev.date().isoformat(), "strategies": {}}
    for name, bt in nets.items():
        b = bt.loc[first:last]
        r = lv.evaluate(b["net"], bench, nw_lags=NW_LAGS)
        r.update(p_one_sided=float(1 - norm.cdf(r["nw_t"])), n_days=len(b),
                 avg_exposure=b["exposure"].mean(), trades=int(b["trade"].sum()), cost=b["cost"].sum(),
                 total_return=float(np.expm1(np.log1p(b["net"]).sum())),
                 bench_total_return=float(np.expm1(bench.loc[first:last].sum())))
        res["strategies"][name] = r
    s = res["strategies"]
    res["H1_C2_pass"] = bool(s["C2"]["p_one_sided"] < ALPHA)
    res["H2_holm"] = lv.holm({k: s[k]["p_one_sided"] for k in ("C1", "C3")}, ALPHA)
    res["run_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print(json.dumps(res, ensure_ascii=False, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "power", "check", "final"])
    ap.add_argument("period", nargs="?", choices=list(FINAL))
    a = ap.parse_args()
    if a.cmd in ("check", "final") and not a.period:
        ap.error("check / final 要指定 A")
    cfg = load_config()
    if a.cmd == "dev":
        cmd_dev(cfg)
    else:
        {"power": cmd_power, "check": cmd_check, "final": cmd_final}[a.cmd](cfg, a.period)


if __name__ == "__main__":
    main()

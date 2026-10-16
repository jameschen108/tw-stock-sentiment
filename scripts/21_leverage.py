"""研究五：槓桿 ETF 對 TAIEX 報酬指數。開發期分析（2003–2023）。

    python scripts/21_leverage.py dev

候選策略與成本設定在 src/pttsent/leverage.py，看結果之前寫死；這裡把它和依賴檔的 sha256 寫進
output/leverage/dev_meta.json。輸出：dev_summary.csv（全期與兩個子期間）、dev_stress.csv（壓力期間）。
1999–2002 只有價格指數，1x 和大盤都少了股息，只當描述，不進判定。
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pttsent import leverage as lv  # noqa: E402
from pttsent.backtest import etf_costs, weights_backtest  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402

DEPS = ["src/pttsent/leverage.py", "src/pttsent/etf.py", "src/pttsent/backtest.py", "docs/leverage/splits.csv"]
PERIODS = {"2003–2023": ("2003-01-02", "2023-12-31"),
           "2003–2014 合成": ("2003-01-02", "2014-12-31"),
           "2015–2023 實際 ETF": ("2015-01-06", "2023-12-31")}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev"])
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "dev":
        cmd_dev(cfg)


if __name__ == "__main__":
    main()

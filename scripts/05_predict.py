"""步驟 5：預測明天漲跌，比較有無情緒特徵，並做扣成本的回測。

    python scripts/05_predict.py            # 開發期（2019–2023）
    python scripts/05_predict.py --final    # 全部定案後，只跑一次最終測試期
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import plots  # noqa: E402
from pttsent.backtest import backtest, summary, trade_costs  # noqa: E402
from pttsent.config import METHODS, load_config, output_path, work_path  # noqa: E402
from pttsent.features import period  # noqa: E402
from pttsent.models import auc_diff_ci, metrics, run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    method = a.method or cfg["sentiment"]["method"]
    tag = "final" if a.final else "dev"
    out = lambda name: output_path(cfg, ticker, method, f"predict_{tag}", name)  # noqa: E731

    daily = pd.read_parquet(work_path(cfg, "processed", f"daily_{ticker}_{method}.parquet"))
    df = period(daily, cfg, a.final)
    probs = run_all(df, cfg["split"]["min_train_days"], cfg["split"]["refit_every"])
    if a.final:
        probs = probs[probs.index >= pd.Timestamp(cfg["split"]["final_test_start"])]
    probs.to_csv(out("predictions.csv"))
    y = df.loc[probs.index, "up_next"]
    print(f"樣本外評估期 {probs.index.min().date()} .. {probs.index.max().date()}（{len(probs)} 天）\n")

    table = pd.DataFrame({m: metrics(y, probs[m]) for m in probs.columns}).T
    table.to_csv(out("metrics.csv"))
    print(table.round(4), "\n")

    by_year = pd.DataFrame({m: ((probs[m] > 0.5) == y).groupby(probs.index.year).mean()
                            for m in probs.columns})
    by_year.to_csv(out("accuracy_by_year.csv"))
    print("各年準確率")
    print(by_year.round(3), "\n")

    ci = {}
    for kind in ("logit", "gbm"):
        pt, lo, hi = auc_diff_ci(y, probs[f"A_price_{kind}"], probs[f"B_price_sent_{kind}"])
        ci[kind] = {"auc_diff": pt, "ci_low": lo, "ci_high": hi}
        print(f"{kind}: AUC(量價+情緒) - AUC(只用量價) = {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    json.dump(ci, open(out("auc_diff.json"), "w"), indent=2)

    buy, sell = trade_costs(cfg)
    ret_next = df.loc[probs.index, "ret_next"]
    curves, rows = {}, {}
    strategies = {"買進持有": pd.Series(1.0, index=probs.index)}
    for m in ["A_price_logit", "B_price_sent_logit", "A_price_gbm", "B_price_sent_gbm"]:
        strategies[m] = (probs[m] > 0.5).astype(float)
    for name, pos in strategies.items():
        bt = backtest(ret_next, pos, buy, sell)
        curves[name] = bt["equity"]
        rows[name] = summary(bt)
    bts = pd.DataFrame(rows).T
    bts.to_csv(out("backtest.csv"))
    print(f"\n回測（買進成本 {buy:.4%}，賣出成本 {sell:.4%}）")
    print(bts.round(4))
    plots.equity_curves(curves, out("equity.png"), f"{ticker} 策略淨值（{tag}）")
    print(f"\n結果 -> {out('x').parent}")


if __name__ == "__main__":
    main()

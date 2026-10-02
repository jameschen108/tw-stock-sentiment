"""預測隔天報酬（ridge、OLS）與漲跌（logit），照預測的正負號做多空，扣成本回測。

    python scripts/13_return_predict.py --ticker 2330 --method classifier_bert
    python scripts/13_return_predict.py ... --target open_to_close   # 隔天開盤到收盤，每天當沖
    python scripts/13_return_predict.py ... --period 2024            # 2024 補充（研究一已用過，非預先登記）

特徵、滾動訓練與研究一相同（A 只用量價、B 加情緒）。策略：
  多空      預測 > 0 做多、< 0 放空（logit 用 機率 - 0.5）
  只做多    預測 < 0 時空手
  多空_門檻 只有 |預測報酬| 超過成本才進場；收盤到收盤時其餘維持原部位，當沖時其餘不交易。
            門檻事先固定：收盤到收盤 = 來回成本 / 2，當沖 = 來回成本；只用在預測報酬的模型。
ridge 的懲罰在訓練窗內交叉驗證，幾乎都選到上限、預測只剩歷史平均；OLS 是不縮減的對照。
成本分現股（放空要融券）與個股期貨兩種，另報損益兩平成本與多空比例相同的隨機基準。
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import plots  # noqa: E402
from pttsent.backtest import (backtest, cost_model, random_sign_pctile, signal_position, summary,  # noqa: E402
                              trade_stats)
from pttsent.config import METHODS, TARGETS, load_config, output_path, target_suffix, work_path  # noqa: E402
from pttsent.features import period  # noqa: E402
from pttsent.models import (MODEL_SETS as MODELS, REG_KINDS as REG, auc_diff_ci, direction,  # noqa: E402
                            metrics, predict_all, regression_metrics)
from pttsent.volume import cw_loss, mean_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--period", choices=["dev", "2024"], default="dev")
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    method = a.method or cfg["sentiment"]["method"]
    suffix = target_suffix(a.target or cfg["target"])
    daytrade = bool(suffix)
    out = lambda name: output_path(cfg, ticker, method, f"return_{a.period}{suffix}", name)  # noqa: E731

    daily = pd.read_parquet(work_path(cfg, "processed", f"daily_{ticker}_{method}{suffix}.parquet"))
    final = a.period == "2024"
    df = period(daily, cfg, final)
    p = predict_all(df, "ret_next", cfg["split"]["min_train_days"], cfg["split"]["refit_every"])
    if final:
        p = p[p.index >= pd.Timestamp(cfg["split"]["final_test_start"])]
    p.to_csv(out("predictions.csv"))
    r = df.loc[p.index, "ret_next"]
    print(f"{ticker} {method} {'開盤到收盤' if daytrade else '收盤到收盤'}："
          f"樣本外 {p.index.min().date()} .. {p.index.max().date()}（{len(p)} 天）\n")

    reg = pd.DataFrame({m: regression_metrics(r, p[m], p["hist_mean"])
                        for m in [f"{k}_{kind}" for kind in REG for k in MODELS] + ["hist_mean"]}).T
    clf = pd.DataFrame({m: metrics(direction(r), p[m], r) for m in ["A_logit", "B_logit"]}).T
    reg.to_csv(out("metrics_reg.csv"))
    clf.to_csv(out("metrics_clf.csv"))
    print("預測報酬（ridge、OLS）")
    print(reg.round(4), "\n")
    print("預測漲跌（logit，平盤日不算）")
    print(clf.round(4), "\n")

    tests = {f"cw_{kind}_B_vs_A": mean_test(cw_loss(r, p[f"A_{kind}"], p[f"B_{kind}"])) for kind in REG}
    auc, lo, hi = auc_diff_ci(direction(r), p["A_logit"], p["B_logit"])
    tests["auc_diff_B_vs_A"] = {"diff": auc, "ci_low": lo, "ci_high": hi}
    json.dump(tests, open(out("tests.json"), "w"), indent=2)
    for kind in REG:
        cw = tests[f"cw_{kind}_B_vs_A"]
        print(f"{kind} B vs A：Clark–West t = {cw['t']:.2f}，單尾 p = {cw['p_one_sided']:.3f}")
    print(f"logit B vs A：AUC 差 {auc:+.4f} [{lo:+.4f}, {hi:+.4f}]\n")

    signals = {f"{k}_{kind}": p[f"{k}_{kind}"] - (0.5 if kind == "logit" else 0.0)
               for k in MODELS for kind in REG + ["logit"]}
    hold_name = "每天做多" if daytrade else "買進持有"   # 當沖時「持有」其實是每天做多一次
    rows, rand, curves = {}, {}, {}
    for inst in ["stock", "futures"]:
        buy, sell, short_fee = cost_model(cfg, inst, daytrade)
        band = (buy + sell) if daytrade else (buy + sell) / 2
        strategies = {hold_name: pd.Series(1.0, index=p.index)}
        if daytrade:
            strategies["每天放空"] = pd.Series(-1.0, index=p.index)
        for name, s in signals.items():
            strategies[f"多空_{name}"] = signal_position(s)
            strategies[f"只做多_{name}"] = signal_position(s, long_only=True)
            if not name.endswith("logit"):
                strategies[f"多空_門檻_{name}"] = signal_position(s, band=band, hold=not daytrade)
        for sname, pos in strategies.items():
            bt = backtest(r, pos, buy, sell, round_trip=daytrade, short_fee=short_fee)
            rows[(inst, sname)] = {**summary(bt), **trade_stats(bt, daytrade)}
            if inst == "stock":
                curves[sname] = bt["equity"]
        print(f"{inst}：買進 {buy:.4%}、賣出 {sell:.4%}、借券費 {short_fee:.4%}，門檻 {band:.4%}")
    for name, s in signals.items():
        rand[name] = random_sign_pctile(r, signal_position(s))
    bts = pd.DataFrame(rows).T
    bts.index.names = ["instrument", "strategy"]
    bts.to_csv(out("backtest.csv"))
    pd.DataFrame(rand).T.to_csv(out("random_sign.csv"))

    show = ["total_return", "sharpe", "max_drawdown", "long_ratio", "short_ratio", "n_trades",
            "breakeven_cost", "avg_cost"]
    print(bts[show].round(4).to_string(), "\n")
    print("多空的毛報酬 vs 多空比例相同的隨機部位（pctile = 贏過幾成的隨機部位）")
    print(pd.DataFrame(rand).T.round(5), "\n")
    keep = [hold_name] + [k for k in curves if k.startswith("多空_") and "門檻" not in k]
    plots.equity_curves({k: curves[k] for k in keep}, out("equity.png"),
                        f"{ticker} 多空策略淨值（現股，{a.period}{suffix}）")
    print(f"結果 -> {out('x').parent}")


if __name__ == "__main__":
    main()

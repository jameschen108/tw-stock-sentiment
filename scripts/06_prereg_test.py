"""步驟 6：預先登記的假設檢定（見 PREREGISTRATION.md）。設定全部寫死，不吃 config 的 ticker／method／target。

    python scripts/06_prereg_test.py            # 開發期（2019–2023），僅供對照
    python scripts/06_prereg_test.py --final    # 最終測試期（2024），只跑一次

需要先跑：步驟 2 --ticker 2603 --method classifier_llm，
          步驟 3 --ticker 2603 --method classifier_llm --target open_to_close
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import analysis  # noqa: E402
from pttsent.backtest import backtest, low_sentiment_days, summary, trade_costs  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402
from pttsent.features import period  # noqa: E402

TICKER, METHOD = "2603", "classifier_llm"
CONTROLS = ["ret", "ret_lag1", "ret_lag2", "mkt_ret", "vol_20", "vlm_z"]   # 同 04_analyze.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    tag = "final" if a.final else "dev"
    daily = pd.read_parquet(work_path(cfg, "processed", f"daily_{TICKER}_{METHOD}_oc.parquet"))
    low_all = low_sentiment_days(daily["sent_mean"])   # 門檻用完整歷史滾動計算，只看過去

    df = period(daily, cfg, a.final)
    if a.final:   # 和 05_predict.py --final 一樣只看 2024 年的列
        df = df[df.index >= pd.Timestamp(cfg["split"]["final_test_start"])]
    low = low_all.reindex(df.index).fillna(False)
    print(f"{TICKER} {METHOD} open_to_close，{tag}：{df.index.min().date()} .. "
          f"{df.index.max().date()}（{len(df)} 天，低情緒日 {int(low.sum())} 天）\n")

    # H1：情緒係數 > 0（單尾）
    res = analysis.ols_hac(df, "ret_next", ["sent", "log_posts"] + CONTROLS)
    coef, p2 = float(res.params["sent"]), float(res.pvalues["sent"])
    p1 = p2 / 2 if coef > 0 else 1 - p2 / 2
    h1 = {"coef": coef, "t": float(res.tvalues["sent"]), "p_one_sided": p1, "n": int(res.nobs),
          "pass": bool(coef > 0 and p1 < 0.05)}

    # H2：低情緒日隔天盤中報酬 < 其他日（單尾 Welch t 檢定）
    r = df["ret_next"]
    lo, other = r[low].dropna(), r[~low].dropna()
    t2 = stats.ttest_ind(lo, other, equal_var=False, alternative="less")
    h2 = {"mean_low_bp": float(lo.mean() * 1e4), "mean_other_bp": float(other.mean() * 1e4),
          "n_low": len(lo), "n_other": len(other), "t": float(t2.statistic),
          "p_one_sided": float(t2.pvalue), "pass": bool(t2.pvalue < 0.05)}

    # H3：放空_低情緒 扣成本後總報酬 > 0
    buy, sell = trade_costs(cfg, daytrade=True)
    s3 = summary(backtest(r, -low.astype(float), buy, sell, round_trip=True))
    h3 = {**{k: float(v) for k, v in s3.items()}, "pass": bool(s3["total_return"] > 0)}

    result = {"period": tag, "H1": h1, "H2": h2, "H3": h3}
    for k in ("H1", "H2", "H3"):
        v = result[k]
        print(k, "通過" if v["pass"] else "未通過",
              {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items() if kk != "pass"})
    out = output_path(cfg, TICKER, METHOD, f"prereg_{tag}", "result.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

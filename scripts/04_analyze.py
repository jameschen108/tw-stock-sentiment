"""步驟 4：分析情緒和股價的關係（描述與統計檢定，不是預測）。

    python scripts/04_analyze.py            # 只用開發期（不含最終測試期）
    python scripts/04_analyze.py --final    # 定案後才加
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import analysis, plots  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402
from pttsent.features import period  # noqa: E402

CONTROLS = ["ret", "ret_lag1", "ret_lag2", "mkt_ret", "vol_20", "vlm_z"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--method", choices=["lexicon", "classifier"])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    method = a.method or cfg["sentiment"]["method"]

    daily = pd.read_parquet(work_path(cfg, "processed", f"daily_{ticker}_{method}.parquet"))
    df = period(daily, cfg, a.final)
    tag = "final" if a.final else "dev"
    out = lambda name: output_path(cfg, ticker, method, f"analysis_{tag}", name)  # noqa: E731
    print(f"期間 {df.index.min().date()} .. {df.index.max().date()}（{len(df)} 天，{tag}）\n")

    xs = ["sent_mean", "bull_ratio", "sent_std", "log_posts", "attn_abn", "push_ratio"]
    corr = analysis.correlations(df, xs, ["ret", "ret_next", "excess_next"])
    corr.to_csv(out("correlations.csv"), index=False)
    print("相關係數（ret = 同一天，ret_next = 隔天）")
    print(corr.pivot(index="x", columns="y", values="pearson").round(3), "\n")

    xc = analysis.cross_correlation(df["sent"], df["ret"])
    xc.to_csv(out("cross_correlation.csv"), index=False)
    plots.cross_correlation(xc, out("cross_correlation.png"), f"{ticker} 情緒與報酬的領先落後")

    gr = analysis.granger(df, "sent", "ret", maxlag=5)
    gr.to_csv(out("granger.csv"), index=False)
    print("Granger 因果檢定 p 值（< 0.05 表示「前者有助於預測後者」）")
    print(gr.pivot(index="lag", columns="cause", values="p").round(4)
          .rename(columns={"sent": "情緒 -> 報酬", "ret": "報酬 -> 情緒"}), "\n")

    res = analysis.ols_hac(df, "ret_next", ["sent", "log_posts"] + CONTROLS)
    (out("regression.txt")).write_text(str(res.summary()))
    print("迴歸：隔天報酬 ~ 情緒 + 關注度 + 控制變數（Newey-West 標準誤）")
    print(pd.DataFrame({"coef": res.params, "t": res.tvalues, "p": res.pvalues})
          .loc[["sent", "log_posts"]].round(5), f"\nR² = {res.rsquared:.4f}\n")

    ev = analysis.event_study(df, "sent_mean")
    ev.to_csv(out("event_study.csv"), index=False)
    plots.event_car(ev, out("event_study.png"), f"{ticker} 情緒極端日前後的累積超額報酬")
    plots.price_vs_sentiment(df, out("price_vs_sentiment.png"), f"{ticker} 股價與 PTT 情緒")
    print(f"表格與圖 -> {out('x').parent}")


if __name__ == "__main__":
    main()

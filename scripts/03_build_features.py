"""步驟 3：把每則情緒彙整成每日特徵，並接上量價與預測目標。

    python scripts/03_build_features.py [--method lexicon|classifier]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import load_config, work_path  # noqa: E402
from pttsent.features import build_daily, daily_sentiment  # noqa: E402
from pttsent.prices import load_stock, load_taiex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--method", choices=["lexicon", "classifier"])
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    method = a.method or cfg["sentiment"]["method"]

    taiex = load_taiex(cfg["data_dir"])
    stock = load_stock(cfg["data_dir"], ticker)
    scored = pd.read_parquet(work_path(cfg, "interim", f"scored_{ticker}_{method}.parquet"))
    sent = daily_sentiment(scored, stock.index, cfg["cutoff"],
                           cfg["sentiment"]["account_weighting"])
    daily = build_daily(stock, taiex, sent, cfg["target"])

    out = work_path(cfg, "processed", f"daily_{ticker}_{method}.parquet")
    daily.to_parquet(out)
    d = daily[(daily.index >= pd.Timestamp(cfg["split"]["start"])) & (daily["n_comments"] > 0)]
    print(f"交易日 {len(daily):,}（{daily.index.min().date()} .. {daily.index.max().date()}）")
    print(f"有 PTT 討論的交易日 {len(d):,}，每日留言中位數 {d['n_comments'].median():.0f}，"
          f"有情緒訊號的帳號中位數 {d['n_polar'].median():.0f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

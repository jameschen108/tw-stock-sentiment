"""步驟 3：把每則情緒彙整成每日特徵，並接上量價與預測目標。

    python scripts/03_build_features.py [--method lexicon|classifier_weak|classifier_llm|classifier_llm_pooled]
    python scripts/03_build_features.py --target open_to_close   # 預測隔天開盤到收盤

open_to_close 時，每日情緒改收到「隔天 09:00 開盤前」為止，盤後到開盤前的討論也算進去；
量價特徵仍只用 t 收盤的資訊，全部都在 t+1 開盤前看得到。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import METHODS, TARGETS, load_config, target_suffix, work_path  # noqa: E402
from pttsent.features import build_daily, daily_sentiment  # noqa: E402
from pttsent.prices import load_stock, load_taiex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--target", choices=TARGETS)
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    method = a.method or cfg["sentiment"]["method"]
    target = a.target or cfg["target"]

    taiex = load_taiex(cfg["data_dir"])
    stock = load_stock(cfg["data_dir"], ticker)
    scored = pd.read_parquet(work_path(cfg, "interim", f"scored_{ticker}_{method}.parquet"))
    if target == "open_to_close":
        # 以 09:00 切日：交易日 d 收到 d 開盤前的貼文；往前挪一列，讓第 t 列＝t+1 開盤前看得到的討論
        sent = daily_sentiment(scored, stock.index, "09:00",
                               cfg["sentiment"]["account_weighting"]).shift(-1)
    else:
        sent = daily_sentiment(scored, stock.index, cfg["cutoff"],
                               cfg["sentiment"]["account_weighting"])
    daily = build_daily(stock, taiex, sent, target)

    out = work_path(cfg, "processed", f"daily_{ticker}_{method}{target_suffix(target)}.parquet")
    daily.to_parquet(out)
    d = daily[(daily.index >= pd.Timestamp(cfg["split"]["start"])) & (daily["n_comments"] > 0)]
    print(f"交易日 {len(daily):,}（{daily.index.min().date()} .. {daily.index.max().date()}）")
    print(f"有 PTT 討論的交易日 {len(d):,}，每日留言中位數 {d['n_comments'].median():.0f}，"
          f"有情緒訊號的帳號中位數 {d['n_polar'].median():.0f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

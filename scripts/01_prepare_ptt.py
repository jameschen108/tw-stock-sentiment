"""步驟 1：PTT jsonl 轉成 parquet（全部年度只需做一次），再挑出討論目標股票的文字。

    python scripts/01_prepare_ptt.py              # 用 config.yaml 的 ticker
    python scripts/01_prepare_ptt.py --ticker 2317
    python scripts/01_prepare_ptt.py --force      # 重新轉檔
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import load_config, work_path  # noqa: E402
from pttsent.ptt import convert_year, select_texts, ticker_aliases, ticker_exclude  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    ptt_dir = work_path(cfg, "interim", "ptt", "x").parent

    for y in cfg["ptt_years"]:
        if (ptt_dir / f"comments_{y}.parquet").exists() and not a.force:
            continue
        t0 = time.time()
        n_art, n_com = convert_year(Path(cfg["data_dir"]) / "pttcc" / f"stock_{y}.jsonl", ptt_dir)
        print(f"{y}: 文章 {n_art:,} 留言 {n_com:,}  ({time.time() - t0:.0f}s)")

    aliases = ticker_aliases(cfg, ticker)
    df = select_texts(ptt_dir, cfg["ptt_years"], ticker, aliases,
                      cfg["sentiment"]["max_comment_lag_days"], ticker_exclude(cfg, ticker))
    out = work_path(cfg, "interim", f"texts_{ticker}.parquet")
    df.to_parquet(out, index=False)
    by_kind = df["kind"].value_counts().to_dict()
    print(f"\n{ticker}（{', '.join(aliases)}）: {len(df):,} 則  {by_kind}")
    print(f"期間 {df['time'].min()} .. {df['time'].max()}  帳號數 {df['account'].nunique():,}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

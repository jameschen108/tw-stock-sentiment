"""步驟 2：幫每則文字打情緒分數。

    python scripts/02_score_sentiment.py                     # 用 config 的方法（預設 lexicon）
    python scripts/02_score_sentiment.py --method classifier_weak  # 需先跑 train_classifier.py --labels weak
    python scripts/02_score_sentiment.py --method classifier_llm   # 需先跑 train_classifier.py --labels llm
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import METHODS, classifier_path, load_config, work_path  # noqa: E402
from pttsent.sentiment import classifier, lexicon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--method", choices=METHODS)
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    method = a.method or cfg["sentiment"]["method"]

    texts = pd.read_parquet(work_path(cfg, "interim", f"texts_{ticker}.parquet"))
    if method == "lexicon":
        texts["score"] = lexicon.score_texts(texts["text"])
    else:
        path = classifier_path(cfg, method.removeprefix("classifier_"), ticker)
        model = classifier.load(path)
        meta = classifier.load_meta(path)
        print(f"分類器 {path.name}（{meta.get('n_labels')} 則標籤，訓練於 {meta.get('trained_at')}）")
        texts["score"] = classifier.score(model, texts["text"])

    out = work_path(cfg, "interim", f"scored_{ticker}_{method}.parquet")
    texts[["uid", "kind", "time", "account", "tag", "score"]].to_parquet(out, index=False)

    for kind, g in texts.groupby("kind"):
        s = g["score"]
        print(f"{kind:8s} n={len(g):>9,}  有情緒訊號 {s.notna().mean():6.1%}  "
              f"偏多 {(s > 0).sum() / max(s.notna().sum(), 1):6.1%}  平均 {np.nanmean(s):+.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""訓練情緒分類器（選用）。訓練完，步驟 2 用 --method classifier 就會改用它。

    python scripts/train_classifier.py --labels weak   # [標的] 文的「分類：多／空」弱標籤，不花錢
    python scripts/train_classifier.py --labels llm    # scripts/llm_label.py collect 產生的標籤

最終測試期（config 的 split.final_test_start 之後）的文字一律排除。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import load_config, work_path  # noqa: E402
from pttsent.sentiment import classifier, lexicon  # noqa: E402
from pttsent.sentiment.llm_label import to_training  # noqa: E402
from pttsent.sentiment.weak_labels import weak_label_dataset  # noqa: E402


def lexicon_baseline(test: pd.DataFrame):
    """同一份測試集上，詞典法的覆蓋率與方向準確率（只看有命中詞的）。"""
    s = lexicon.score_texts(test["text"])
    has = ~np.isnan(s) & (s != 0) & test["label"].isin(["bullish", "bearish"]).to_numpy()
    pred = np.where(s[has] > 0, "bullish", "bearish")
    acc = (pred == test["label"].to_numpy()[has]).mean() if has.any() else np.nan
    return has.mean(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["weak", "llm"], required=True)
    ap.add_argument("--ticker")
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])

    if a.labels == "weak":
        ds = weak_label_dataset(work_path(cfg, "interim", "ptt", "x").parent, cfg["ptt_years"])
    else:
        ds = to_training(pd.read_parquet(work_path(cfg, "llm", f"labels_{ticker}.parquet")))
    # 最終測試期的文字不進訓練，也不拿來看測試成績，否則開發時就先看過 2024 了
    final_start = pd.Timestamp(cfg["split"]["final_test_start"])
    n_all = len(ds)
    ds = ds[ds["time"] < final_start].reset_index(drop=True)
    print(f"排除 {final_start.date()} 之後的 {n_all - len(ds):,} 則（最終測試期）")
    print(f"標籤 {len(ds):,} 則：{ds['label'].value_counts().to_dict()}")

    model, report = classifier.train(ds)
    _, test = classifier.split_by_time(ds)
    print(f"\n時間切分的測試集（{test['time'].min().date()} 之後，{len(test):,} 則）")
    print(report)
    cov, acc = lexicon_baseline(test)
    print(f"對照：詞典法在同一測試集只判得出 {cov:.1%} 的文字，其中方向準確率 {acc:.1%}")

    path = cfg["sentiment"]["classifier_path"]
    classifier.save(model, path)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()

"""訓練情緒分類器（選用）。訓練完，步驟 2 用 --method classifier_weak 或 classifier_llm 就會改用它。

    python scripts/train_classifier.py --labels weak   # [標的] 文的「分類：多／空」弱標籤，不花錢
    python scripts/train_classifier.py --labels llm    # scripts/llm_label.py collect 產生的標籤
    python scripts/train_classifier.py --labels llm_pooled   # config 的 llm.pool_tickers 合併訓練一個
    python scripts/train_classifier.py --labels llm_pooled --model bert   # 同上，改用 BERT 微調（約 30 分鐘）

最終測試期（config 的 split.final_test_start 之後）的文字一律排除。
模型存到 data/models/sentiment_clf_{weak | llm_代號 | llm_pooled}.joblib，同名 .json 是訓練紀錄。
之後步驟 2–5 用 --method classifier_weak、classifier_llm 或 classifier_llm_pooled。
llm_pooled 另外列出每檔股票的測試成績，並和「只用該股票自己的標籤、同一個時間切點」訓練的模型比較；
config 的 llm.test_tickers（例如 2317）的標籤不參與訓練，只拿來測試最終模型。
BERT 存成資料夾 data/models/sentiment_bert_llm_pooled/，步驟 2–5 用 --method classifier_bert。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import bert_path, classifier_path, load_config, work_path  # noqa: E402
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


def pooled_vs_own(ds: pd.DataFrame, test: pd.DataFrame) -> dict:
    """每檔股票的測試成績：合併模型 vs 只用該股票自己的標籤（同一個時間切點，測試文字完全相同）。"""
    train = ds[ds["time"] < test["time"].min()]
    out, rows = {}, {}
    for t, te in test.groupby("ticker"):
        tr = train[train["ticker"] == t]
        own = te.assign(pred=classifier.build_pipeline().fit(tr["text"], tr["label"]).predict(te["text"]))
        out[t] = {"n_test": len(te), "pooled": classifier.report(te), "own_only": classifier.report(own)}
        for name in ("pooled", "own_only"):
            r = out[t][name]
            rows[(t, name)] = {c: r[c]["f1-score"] for c in ("bearish", "bullish", "neutral", "irrelevant")} \
                | {"macro": r["macro avg"]["f1-score"]}
    print("\n各股票 F1：合併 vs 只用自己的標籤（同一測試集）")
    print(pd.DataFrame(rows).T.round(3))
    return out


def train_bert(ds: pd.DataFrame):
    """和 classifier.train 相同的時間切分；訓練段最後 10% 當驗證集挑 epoch，再用全部資料、同樣 epoch 數重訓。"""
    from pttsent.sentiment import bert
    tr, te = classifier.split_by_time(ds)
    fit, valid = classifier.split_by_time(tr, test_frac=0.1)
    model, tok, history, best_epoch = bert.fine_tune(fit, valid)
    test = te.assign(pred=bert.predict(model, tok, te["text"], bert.titles_of(te)))
    print(f"最好的 epoch：{best_epoch}（驗證集 macro-F1 {[round(h, 3) for h in history]}）；用全部 {len(ds):,} 則重訓")
    model, tok, _, _ = bert.fine_tune(ds, epochs=best_epoch)
    predict = lambda df: bert.predict(model, tok, df["text"], bert.titles_of(df))  # noqa: E731
    return (model, tok), test, predict, {"best_epoch": best_epoch, "valid_macro_f1": history}


def held_out(cfg, predict, final_start) -> dict:
    """config 的 llm.test_tickers：沒參與訓練的股票，拿來測最終模型。"""
    out = {}
    for t in cfg["llm"].get("test_tickers", []):
        path = work_path(cfg, "llm", f"labels_{t}.parquet")
        if not path.exists():
            continue
        te = to_training(pd.read_parquet(path))
        te = te[te["time"] < final_start]
        te = te.assign(pred=predict(te))
        out[t] = {"n_test": len(te), "all": classifier.report(te),
                  "comments": classifier.report(te[te["kind"] == "comment"])}
        print(f"\n沒參與訓練的 {t}（{len(te)} 則）")
        print(pd.DataFrame(out[t]["all"]).T.round(3))
        print(f"只看留言 macro-F1 {out[t]['comments']['macro avg']['f1-score']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["weak", "llm", "llm_pooled"], required=True)
    ap.add_argument("--model", choices=["tfidf", "bert"], default="tfidf")
    ap.add_argument("--ticker")
    a = ap.parse_args()
    if a.model == "bert" and a.labels != "llm_pooled":
        ap.error("--model bert 目前只支援 --labels llm_pooled")
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])

    if a.labels == "weak":
        ds = weak_label_dataset(work_path(cfg, "interim", "ptt", "x").parent, cfg["ptt_years"])
    elif a.labels == "llm":
        ds = to_training(pd.read_parquet(work_path(cfg, "llm", f"labels_{ticker}.parquet")))
    else:
        ds = pd.concat([to_training(pd.read_parquet(work_path(cfg, "llm", f"labels_{t}.parquet"))).assign(ticker=t)
                        for t in cfg["llm"]["pool_tickers"]], ignore_index=True)
    # 最終測試期的文字不進訓練，也不拿來看測試成績，否則開發時就先看過 2024 了
    final_start = pd.Timestamp(cfg["split"]["final_test_start"])
    n_all = len(ds)
    ds = ds[ds["time"] < final_start].reset_index(drop=True)
    print(f"排除 {final_start.date()} 之後的 {n_all - len(ds):,} 則（最終測試期）")
    print(f"標籤 {len(ds):,} 則：{ds['label'].value_counts().to_dict()}")

    if "ticker" in ds:
        print(f"股票：{ds['ticker'].value_counts().to_dict()}")

    if a.model == "bert":
        model, test, predict, bert_info = train_bert(ds)
    else:
        model, test = classifier.train(ds)
        predict = lambda df: model.predict(df["text"])  # noqa: E731
        bert_info = {}
    report = classifier.report(test)
    print(f"\n時間切分的測試集（{test['time'].min().date()} 之後，{len(test):,} 則）")
    print(pd.DataFrame(report).T.round(3))
    cov, acc = lexicon_baseline(test)
    print(f"對照：詞典法在同一測試集只判得出 {cov:.1%} 的文字，其中方向準確率 {acc:.1%}")
    by_ticker = pooled_vs_own(ds, test) if "ticker" in ds and a.model == "tfidf" else None
    held = held_out(cfg, predict, final_start) if a.labels == "llm_pooled" else {}

    meta = {
        "labels": a.labels,
        "model": a.model,
        "ticker": ticker if a.labels == "llm" else None,
        **({"tickers": sorted(ds["ticker"].unique())} if "ticker" in ds else {}),
        "data_period": [ds["time"].min(), ds["time"].max()],
        "excluded_from": final_start,
        "n_labels": len(ds),
        "label_counts": ds["label"].value_counts().to_dict(),
        "test_period": [test["time"].min(), test["time"].max()],
        "test_report": report,
        **({"test_by_ticker": by_ticker} if by_ticker else {}),
        **({"held_out_tickers": held} if held else {}),
        **bert_info,
        "lexicon_baseline": {"coverage": cov, "accuracy": acc},
        "note": "模型最後用全部 n_labels 則重訓；test_report 是只用前 80% 訓練時在後 20% 的成績",
    }
    if a.model == "bert":
        from pttsent.sentiment import bert
        path = bert_path(cfg)
        bert.save(*model, path, meta)
        print(f"\n-> {path}/（training.json 是訓練紀錄）")
    else:
        path = classifier_path(cfg, a.labels, ticker)
        classifier.save(model, path, meta)
        print(f"\n-> {path}")
        print(f"-> {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()

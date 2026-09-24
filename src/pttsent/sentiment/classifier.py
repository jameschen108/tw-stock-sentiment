"""輕量分類器：字元 n-gram TF-IDF + logistic regression。

訓練資料可以是 [標的] 弱標籤，或 LLM 標註（見 llm_label.py）。
分數 = P(看多) - P(看空)；若判定為 irrelevant 則為 NaN。
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                                  max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)),
    ])


def split_by_time(df: pd.DataFrame, test_frac: float = 0.2):
    """依時間切：前 80% 訓練、後 20% 測試（不隨機打亂，避免同一時期的文章兩邊都有）。"""
    df = df.sort_values("time").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def train(df: pd.DataFrame, test_frac: float = 0.2):
    tr, te = split_by_time(df, test_frac)
    model = build_pipeline().fit(tr["text"], tr["label"])
    report = classification_report(te["label"], model.predict(te["text"]), digits=3)
    model = build_pipeline().fit(df["text"], df["label"])   # 報告完再用全部資料重訓
    return model, report


def score(model, texts) -> np.ndarray:
    proba = model.predict_proba(list(texts))
    classes = list(model.classes_)
    col = {c: proba[:, i] for i, c in enumerate(classes)}
    zero = np.zeros(len(proba))
    s = col.get("bullish", zero) - col.get("bearish", zero)
    if "irrelevant" in classes:
        s = np.where(proba.argmax(axis=1) == classes.index("irrelevant"), np.nan, s)
    return s


def save(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)

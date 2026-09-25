"""輕量分類器：字元 n-gram TF-IDF + logistic regression。

訓練資料可以是 [標的] 弱標籤，或 LLM 標註（見 llm_label.py）。
分數 = P(看多) - P(看空)；若判定為 irrelevant 則為 NaN。
模型存成 .joblib，旁邊的同名 .json 記錄標籤來源、訓練期間、測試成績與 scikit-learn 版本。
"""
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
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
    """回傳（用全部資料重訓的模型, 時間切分的測試集，多一欄 pred 是只用前段訓練時的預測）。"""
    tr, te = split_by_time(df, test_frac)
    model = build_pipeline().fit(tr["text"], tr["label"])
    te = te.assign(pred=model.predict(te["text"]))
    model = build_pipeline().fit(df["text"], df["label"])   # 評估完再用全部資料重訓
    return model, te


def report(test: pd.DataFrame) -> dict:
    return classification_report(test["label"], test["pred"], output_dict=True, zero_division=0)


def score(model, texts) -> np.ndarray:
    proba = model.predict_proba(list(texts))
    classes = list(model.classes_)
    col = {c: proba[:, i] for i, c in enumerate(classes)}
    zero = np.zeros(len(proba))
    s = col.get("bullish", zero) - col.get("bearish", zero)
    if "irrelevant" in classes:
        s = np.where(proba.argmax(axis=1) == classes.index("irrelevant"), np.nan, s)
    return s


def save(model, path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = {**meta, "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")


def load_meta(path) -> dict:
    p = path.with_suffix(".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load(path):
    """只讀自己訓練的檔：joblib 讀檔時會執行檔案裡的程式碼。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先跑 scripts/train_classifier.py")
    trained_with = load_meta(path).get("sklearn_version")
    if trained_with is None:
        warnings.warn(f"{path.name} 沒有訓練紀錄（.json），不確定是用什麼資料訓練的")
    elif trained_with != sklearn.__version__:
        warnings.warn(f"{path.name} 是用 scikit-learn {trained_with} 訓練的，"
                      f"目前是 {sklearn.__version__}，結果可能不同；建議重訓")
    return joblib.load(path)

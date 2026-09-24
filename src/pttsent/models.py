"""預測明天漲跌：滾動式（walk-forward）訓練，比較「只用量價」與「量價 + 情緒」。

每 refit_every 天重新訓練一次，訓練資料只用預測日之前的列。
第 t 列的標籤是 t+1 的報酬，在 t+1 收盤才知道；在 t 收盤做預測時，
訓練集最多到 t-1 列（它的標籤在 t 收盤已經實現），所以不會用到未來。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PRICE_FEATURES, SENT_FEATURES

FEATURE_SETS = {"A_price": PRICE_FEATURES, "B_price_sent": PRICE_FEATURES + SENT_FEATURES}


def make_model(kind: str):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


def walk_forward(df: pd.DataFrame, features, kind: str, target: str = "up_next",
                 min_train: int = 250, refit_every: int = 21) -> pd.DataFrame:
    d = df.dropna(subset=list(features))
    preds = []
    for start in range(min_train, len(d), refit_every):
        train = d.iloc[:start].dropna(subset=[target])
        test = d.iloc[start:start + refit_every]
        assert train.index.max() < test.index.min()
        if train[target].nunique() < 2:
            continue
        m = make_model(kind).fit(train[features], train[target])
        preds.append(pd.DataFrame({"prob": m.predict_proba(test[features])[:, 1],
                                   "train_end": train.index.max()}, index=test.index))
    return pd.concat(preds)


def baselines(df: pd.DataFrame, target: str = "up_next") -> pd.DataFrame:
    """always_up：用過去上漲比例當機率（實際上幾乎永遠猜漲）；momentum：今天漲就猜明天漲。"""
    past_rate = df[target].shift(1).expanding(20).mean()
    return pd.DataFrame({"always_up": past_rate,
                         "momentum": (df["ret"] > 0).astype(float)}, index=df.index)


def metrics(y: pd.Series, prob: pd.Series) -> dict:
    m = pd.concat([y, prob], axis=1, keys=["y", "p"]).dropna()
    out = {"n": len(m), "accuracy": accuracy_score(m["y"], m["p"] > 0.5),
           "up_rate": m["y"].mean()}
    if m["p"].nunique() > 2:
        out["auc"] = roc_auc_score(m["y"], m["p"])
        out["brier"] = brier_score_loss(m["y"], m["p"].clip(0, 1))
    return out


def run_all(df: pd.DataFrame, min_train: int, refit_every: int) -> pd.DataFrame:
    """回傳每天各模型的預測機率（欄位 = 模型名稱）。"""
    probs = baselines(df)
    for kind in ("logit", "gbm"):
        for name, feats in FEATURE_SETS.items():
            wf = walk_forward(df, feats, kind, min_train=min_train, refit_every=refit_every)
            probs[f"{name}_{kind}"] = wf["prob"]
    first_oos = probs.drop(columns=["always_up", "momentum"]).dropna(how="all").index.min()
    return probs[probs.index >= first_oos]


def auc_diff_ci(y, p_a, p_b, block: int = 20, reps: int = 1000, seed: int = 0):
    """AUC(B) - AUC(A) 的區塊 bootstrap 95% 信賴區間（保留時間相依）。"""
    m = pd.concat([y, p_a, p_b], axis=1, keys=["y", "a", "b"]).dropna().to_numpy()
    n = len(m)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        starts = rng.integers(0, n - block, size=n // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        s = m[idx]
        if len(np.unique(s[:, 0])) < 2:
            continue
        diffs.append(roc_auc_score(s[:, 0], s[:, 2]) - roc_auc_score(s[:, 0], s[:, 1]))
    point = roc_auc_score(m[:, 0], m[:, 2]) - roc_auc_score(m[:, 0], m[:, 1])
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

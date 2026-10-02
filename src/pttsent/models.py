"""預測明天漲跌（logit）與報酬（ridge）：滾動式（walk-forward）訓練，比較「只用量價」與「量價 + 情緒」。

每 refit_every 天重新訓練一次，訓練資料只用預測日之前的列。
第 t 列的標籤是 t+1 的報酬，在 t+1 收盤才知道；在 t 收盤做預測時，
訓練集最多到 t-1 列（它的標籤在 t 收盤已經實現），所以不會用到未來。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PRICE_FEATURES, SENT_FEATURES

FEATURE_SETS = {"A_price": PRICE_FEATURES, "B_price_sent": PRICE_FEATURES + SENT_FEATURES}
MODEL_SETS = {"A": "A_price", "B": "B_price_sent"}
REG_KINDS = ["ridge", "ols"]


def make_model(kind: str):
    """C=0.01：C=0.1 時三檔股票的 Brier 都比「過去上漲比例」這個常數差，等於在擬合雜訊。
    梯度提升樹拿掉了：同樣的資料上 Brier 約 0.27，過擬合更嚴重。"""
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
    if kind == "ridge":
        # 懲罰強度在訓練窗內用時間序列交叉驗證選，不會看到預測日之後的資料
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(0, 6, 13), cv=TimeSeriesSplit(5)))
    if kind == "ols":
        # ridge 的交叉驗證幾乎都選到懲罰上限，預測只剩歷史平均、永遠 > 0；OLS 不縮減，正負號才會變
        return make_pipeline(StandardScaler(), LinearRegression())
    raise ValueError(kind)


def direction(ret: pd.Series) -> pd.Series:
    """漲 = 1、跌 = 0、平盤或缺值 = NaN（平盤日不放進訓練，免得「不漲」被平盤日灌水）。"""
    return (ret > 0).astype(float).where((ret != 0) & ret.notna())


def walk_forward(df: pd.DataFrame, features, kind: str, target: str = "up_next",
                 min_train: int = 250, refit_every: int = 21, clip: float = 0.01) -> pd.DataFrame:
    """logit 回傳欄位 prob（上漲機率），ridge / ols 回傳 pred（預測報酬）。

    迴歸的訓練目標在訓練窗內的 clip / 1 - clip 分位數截尾，避免少數極端日主導平方誤差。
    """
    d = df.dropna(subset=list(features))
    preds = []
    for start in range(min_train, len(d), refit_every):
        train = d.iloc[:start].dropna(subset=[target])
        test = d.iloc[start:start + refit_every]
        assert train.index.max() < test.index.min()
        if kind in ("ridge", "ols"):
            y = train[target].clip(*train[target].quantile([clip, 1 - clip]))
            m = make_model(kind).fit(train[features], y)
            out = {"pred": m.predict(test[features])}
        else:
            if train[target].nunique() < 2:
                continue
            m = make_model(kind).fit(train[features], train[target])
            out = {"prob": m.predict_proba(test[features])[:, 1]}
        preds.append(pd.DataFrame({**out, "train_end": train.index.max()}, index=test.index))
    return pd.concat(preds)


def baselines(df: pd.DataFrame, target: str = "up_next") -> pd.DataFrame:
    """always_up：用過去上漲比例當機率（實際上幾乎永遠猜漲）；momentum：今天漲就猜明天漲。"""
    past_rate = df[target].shift(1).expanding(20).mean()
    return pd.DataFrame({"always_up": past_rate,
                         "momentum": (df["ret"] > 0).astype(float)}, index=df.index)


def metrics(y: pd.Series, prob: pd.Series, ret: pd.Series) -> dict:
    """ret 是隔天報酬（連續值）。ic = Spearman(預測機率, 隔天報酬)：AUC 只看漲跌，ic 也用到漲跌幅。"""
    m = pd.concat([y, prob, ret], axis=1, keys=["y", "p", "r"]).dropna()
    out = {"n": len(m), "accuracy": accuracy_score(m["y"], m["p"] > 0.5),
           "up_rate": m["y"].mean()}
    if m["p"].nunique() > 2:
        out["auc"] = roc_auc_score(m["y"], m["p"])
        out["brier"] = brier_score_loss(m["y"], m["p"].clip(0, 1))
        out["ic"], out["ic_p"] = spearmanr(m["p"], m["r"])
    return out


def run_all(df: pd.DataFrame, min_train: int, refit_every: int) -> pd.DataFrame:
    """回傳每天各模型的預測機率（欄位 = 模型名稱）。"""
    probs = baselines(df)
    for name, feats in FEATURE_SETS.items():
        wf = walk_forward(df, feats, "logit", min_train=min_train, refit_every=refit_every)
        probs[f"{name}_logit"] = wf["prob"]
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


def hist_mean(ret_next: pd.Series, min_periods: int = 20) -> pd.Series:
    """迴歸的基準：第 t 列只用已實現的報酬（到 t-1 列的 ret_next）算平均。"""
    return ret_next.shift(1).expanding(min_periods).mean()


def regression_metrics(y: pd.Series, pred: pd.Series, bench: pd.Series) -> dict:
    """y 是隔天報酬，bench 是歷史平均的預測。

    r2_os_mean：相對歷史平均的樣本外 R²（Campbell–Thompson）；r2_os_zero：相對「預測 0」。
    個股的歷史平均本身就很吵，所以兩個都報。sign_acc 不算平盤日。
    """
    m = pd.concat([y, pred, bench], axis=1, keys=["y", "p", "b"]).dropna()
    sse = ((m["y"] - m["p"]) ** 2).sum()
    nz = m[m["y"] != 0]
    ic, ic_p = spearmanr(m["p"], m["y"])
    return {"n": len(m), "mse": sse / len(m),
            "r2_os_mean": 1 - sse / ((m["y"] - m["b"]) ** 2).sum(),
            "r2_os_zero": 1 - sse / (m["y"] ** 2).sum(),
            "ic": ic, "ic_p": ic_p,
            "sign_acc": (np.sign(nz["p"]) == np.sign(nz["y"])).mean(),
            "pred_pos": (m["p"] > 0).mean(), "pred_sd": m["p"].std()}


def predict_all(df: pd.DataFrame, target: str = "ret_next", min_train: int = 250,
                refit_every: int = 21) -> pd.DataFrame:
    """A/B × ridge、OLS（預測 target）與 logit（預測 target 的正負，平盤不訓練）。

    只留所有模型都有預測的列，另附歷史平均（迴歸的基準）。欄位：A_ridge、A_ols、A_logit、B_…、hist_mean。
    """
    df = df.assign(_dir=direction(df[target]))
    kw = dict(min_train=min_train, refit_every=refit_every)
    cols = {}
    for k, name in MODEL_SETS.items():
        feats = FEATURE_SETS[name]
        for kind in REG_KINDS:
            cols[f"{k}_{kind}"] = walk_forward(df, feats, kind, target=target, **kw)["pred"]
        cols[f"{k}_logit"] = walk_forward(df, feats, "logit", target="_dir", **kw)["prob"]
    out = pd.DataFrame(cols).dropna()
    out["hist_mean"] = hist_mean(df[target]).reindex(out.index)
    return out

"""「情緒對股價有沒有影響」的統計分析：相關、領先落後、Granger、迴歸、事件研究。"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


def correlations(df: pd.DataFrame, xs, ys) -> pd.DataFrame:
    rows = []
    for x in xs:
        for y in ys:
            d = df[[x, y]].dropna()
            rows.append({"x": x, "y": y, "n": len(d),
                         "pearson": d[x].corr(d[y]),
                         "spearman": d[x].corr(d[y], method="spearman")})
    return pd.DataFrame(rows)


def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """corr(x_t, y_{t+k})。k > 0：情緒領先報酬；k < 0：報酬領先情緒。"""
    return pd.DataFrame([{"lag": k, "corr": x.corr(y.shift(-k))}
                         for k in range(-max_lag, max_lag + 1)])


def granger(df: pd.DataFrame, x: str, y: str, maxlag: int = 5) -> pd.DataFrame:
    """兩個方向都測：x -> y 以及 y -> x。p 值來自 SSR F 檢定。"""
    rows = []
    for cause, effect in [(x, y), (y, x)]:
        d = df[[effect, cause]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(d, maxlag=maxlag, verbose=False)
        for lag, (tests, _) in res.items():
            rows.append({"cause": cause, "effect": effect, "lag": lag,
                         "F": tests["ssr_ftest"][0], "p": tests["ssr_ftest"][1]})
    return pd.DataFrame(rows)


def ols_hac(df: pd.DataFrame, y: str, xs, maxlags: int = 5):
    """OLS + Newey-West 標準誤（報酬有自相關與異質變異時，一般標準誤會太樂觀）。"""
    d = df[[y] + list(xs)].dropna()
    X = sm.add_constant(d[list(xs)])
    return sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def event_study(df: pd.DataFrame, signal: str, q: float = 0.1,
                window=(-5, 10), min_history: int = 60) -> pd.DataFrame:
    """情緒極端日前後的平均累積超額報酬（個股 - 大盤）。

    「極端」用擴張視窗分位數判定，只用當天以前的資料，不會偷看未來的分布。
    """
    s = df[signal]
    hi_cut = s.shift(1).expanding(min_history).quantile(1 - q)
    lo_cut = s.shift(1).expanding(min_history).quantile(q)
    ar = (df["ret"] - df["mkt_ret"]).to_numpy()
    idx = {"high": np.where(s > hi_cut)[0], "low": np.where(s < lo_cut)[0]}
    offsets = range(window[0], window[1] + 1)
    rows = []
    for name, pos in idx.items():
        pos = pos[(pos + window[0] >= 0) & (pos + window[1] < len(df))]
        if not len(pos):
            continue
        mat = np.stack([ar[p + window[0]: p + window[1] + 1] for p in pos])
        # 以事件日前一天為 0 累積
        car = np.nancumsum(mat, axis=1) - np.nancumsum(mat, axis=1)[:, [-window[0] - 1]]
        for j, k in enumerate(offsets):
            rows.append({"group": name, "offset": k, "car": np.nanmean(car[:, j]),
                         "se": np.nanstd(car[:, j]) / np.sqrt(len(pos)), "n_events": len(pos)})
    return pd.DataFrame(rows)

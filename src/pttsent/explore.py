"""探索方向的工具：週頻資料、情緒極端日、極端日的報酬差檢定、對沖後的超額報酬。

設定與主要檢定見 docs/explore_directions.md。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import PRICE_FEATURES, SENT_FEATURES


def weekly(daily: pd.DataFrame, start, end) -> pd.DataFrame:
    """每週一列，索引是該週最後一個交易日（在這天收盤決定下週的部位）。

    量價特徵取該週最後一天那一列，情緒特徵取該週每天的平均；
    ret_next = 下週每天 ret 的加總，下週有任何一天缺值就是缺值。
    只留決策日 >= start、且下週最後一天 < end 的週，所以 end 之後的報酬不會出現。
    """
    wk = daily.index.to_period("W-FRI")
    last_day = daily.index.to_series().groupby(wk).max()
    out = daily.loc[last_day.to_numpy(), PRICE_FEATURES].set_axis(last_day.index)
    out = out.join(daily[SENT_FEATURES].groupby(wk).mean())
    wret = daily["ret"].groupby(wk).agg(lambda s: s.sum(skipna=False))
    out["ret_next"] = wret.shift(-1)
    out["target_end"] = last_day.shift(-1)
    out.index = pd.DatetimeIndex(last_day.to_numpy(), name="date")
    keep = (out.index >= pd.Timestamp(start)) & (out["target_end"] < pd.Timestamp(end)).to_numpy()
    return out[keep]


def sentiment_extremes(sent_mean: pd.Series, window: int = 250, q: float = 0.1,
                       min_periods: int = 60) -> tuple[pd.Series, pd.Series]:
    """(悲觀日, 樂觀日)：情緒落在過去 window 個交易日（不含當天）的最低 / 最高 q 比例。

    門檻只用當天以前的資料；沒有情緒訊號的日子兩者都是 False。
    """
    past = sent_mean.shift(1).rolling(window, min_periods=min_periods)
    has = sent_mean.notna()
    return (sent_mean <= past.quantile(q)) & has, (sent_mean >= past.quantile(1 - q)) & has


def extreme_position(low: pd.Series, high: pd.Series, hold: int = 1) -> pd.Series:
    """反向：悲觀日收盤做多、樂觀日收盤放空，持有 hold 天；持有期間出現新的極端日，以最新的為準。"""
    sig = pd.Series(np.nan, index=low.index)
    sig[low.to_numpy()] = 1.0
    sig[high.to_numpy()] = -1.0
    if hold > 1:
        sig = sig.ffill(limit=hold - 1)
    return sig.fillna(0.0)


def forward_return(ret: pd.Series, h: int) -> pd.Series:
    """第 t 列 = t+1 到 t+h 的對數報酬加總（任何一天缺值就是缺值）。"""
    return sum(ret.shift(-k) for k in range(1, h + 1))


def extreme_spread(fwd: pd.Series, low: pd.Series, high: pd.Series, controls: pd.DataFrame,
                   maxlags: int) -> dict:
    """fwd ~ 悲觀日 + 樂觀日 + controls（OLS，Newey–West），檢定 悲觀 − 樂觀 的係數差，雙尾。

    另附不加控制變數的平均差（raw_diff），對照控制量價前後的差別。
    """
    d = pd.concat([fwd.rename("y"), low.astype(float).rename("low"),
                   high.astype(float).rename("high"), controls], axis=1).dropna()
    m = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    t = m.t_test("low - high = 0")
    lo, hi = d.loc[d["low"] == 1, "y"], d.loc[d["high"] == 1, "y"]
    return {"n": len(d), "n_low": len(lo), "n_high": len(hi),
            "raw_diff": lo.mean() - hi.mean(), "diff": float(np.squeeze(t.effect)),
            "t": float(np.squeeze(t.tvalue)), "p": float(np.squeeze(t.pvalue))}


def hedged_return(ret_next: pd.Series, mkt_next: pd.Series) -> pd.Series:
    """做多個股、放空同金額的大盤（beta = 1）一天的報酬，以對數表示，方便丟進 backtest。"""
    return np.log1p(np.expm1(ret_next) - np.expm1(mkt_next))

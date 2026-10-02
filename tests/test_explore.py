import numpy as np
import pandas as pd

from pttsent.explore import (extreme_position, extreme_spread, forward_return, hedged_return,
                             sentiment_extremes, weekly)
from pttsent.features import PRICE_FEATURES, SENT_FEATURES


def _daily(days):
    df = pd.DataFrame(0.0, index=days, columns=PRICE_FEATURES + SENT_FEATURES)
    df["ret"] = np.arange(len(days), dtype=float)
    df["sent"] = np.arange(len(days), dtype=float)
    return df


def test_weekly_target_is_next_week_and_never_crosses_end():
    days = pd.bdate_range("2023-12-04", "2024-01-12").drop(pd.Timestamp("2023-12-29"))   # 週五休市
    w = weekly(_daily(days), "2023-12-01", "2024-01-01")
    # 決策日：12/8、12/15、12/22；12/28 那週的下一週在 2024，不能出現
    assert list(w.index) == list(pd.to_datetime(["2023-12-08", "2023-12-15", "2023-12-22"]))
    d = _daily(days)
    assert w.loc["2023-12-08", "ret_next"] == d.loc["2023-12-11":"2023-12-15", "ret"].sum()
    assert w.loc["2023-12-22", "ret_next"] == d.loc["2023-12-25":"2023-12-28", "ret"].sum()
    assert w.loc["2023-12-08", "ret"] == d.loc["2023-12-08", "ret"]          # 量價取最後一天
    assert w.loc["2023-12-08", "sent"] == d.loc["2023-12-04":"2023-12-08", "sent"].mean()   # 情緒取平均


def test_weekly_target_missing_if_any_day_missing():
    days = pd.bdate_range("2023-11-06", "2023-11-24")
    d = _daily(days)
    d.loc["2023-11-15", "ret"] = np.nan
    w = weekly(d, "2023-11-01", "2024-01-01")
    assert np.isnan(w.loc["2023-11-10", "ret_next"]) and w.loc["2023-11-17", "ret_next"] == d.loc["2023-11-20":, "ret"].sum()


def test_sentiment_extremes_only_use_the_past():
    s = pd.Series([0.0, 1.0, 2.0, 3.0, -5.0, 9.0, np.nan])
    low, high = sentiment_extremes(s, window=4, q=0.25, min_periods=3)
    assert low.tolist() == [False, False, False, False, True, False, False]
    assert high.tolist() == [False, False, False, True, False, True, False]


def test_extreme_position_holds_and_latest_wins():
    low = pd.Series([True, False, False, False, False, False])
    high = pd.Series([False, False, True, False, False, False])
    assert extreme_position(low, high, hold=1).tolist() == [1, 0, -1, 0, 0, 0]
    assert extreme_position(low, high, hold=3).tolist() == [1, 1, -1, -1, -1, 0]


def test_forward_return_sums_next_h_days():
    r = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert forward_return(r, 2).iloc[:2].tolist() == [5.0, 7.0] and forward_return(r, 2).iloc[2:].isna().all()


def test_extreme_spread_recovers_known_difference():
    rng = np.random.default_rng(0)
    n = 2000
    low = pd.Series(rng.random(n) < 0.1)
    high = pd.Series(~low & (rng.random(n) < 0.1))
    y = 0.01 * low - 0.01 * high + rng.normal(0, 0.001, n)
    r = extreme_spread(y, low, high, pd.DataFrame({"x": rng.normal(size=n)}), maxlags=5)
    assert abs(r["diff"] - 0.02) < 0.001 and r["p"] < 0.001


def test_hedged_return_is_stock_minus_market():
    h = hedged_return(pd.Series(np.log([1.03])), pd.Series(np.log([1.01])))
    assert np.isclose(np.expm1(h.iloc[0]), 0.02)

import numpy as np
import pandas as pd

from pttsent.backtest import backtest, low_sentiment_days, summary


def test_short_daytrade_pays_round_trip_each_day():
    oc = pd.Series(np.log([0.98, 1.01, 1.00]))          # 開盤到收盤：-2%、+1%、0%
    bt = backtest(oc, pd.Series([-1.0, -1.0, 0.0]), 0.001, 0.002, round_trip=True)
    assert np.allclose(bt["gross"], [0.02, -0.01, 0.0])
    assert np.allclose(bt["cost"], [0.003, 0.003, 0.0])
    s = summary(bt)
    assert s["n_trades"] == 2 and np.isclose(s["exposure"], 2 / 3)


def test_low_sentiment_days_only_uses_past():
    s = pd.Series([0.0, 1.0, 2.0, 3.0, -1.0, np.nan, 5.0])
    low = low_sentiment_days(s, window=4, q=0.5, min_periods=3)
    # 第 3 天門檻 = 過去 [0,1,2] 的中位數 1 -> 3 不算低；第 4 天門檻 = [0,1,2,3] 的 1.5 -> -1 算低
    assert low.tolist() == [False, False, False, False, True, False, False]

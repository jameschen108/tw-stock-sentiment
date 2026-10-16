import numpy as np
import pandas as pd

from pttsent.backtest import (backtest, low_sentiment_days, random_sign_pctile, signal_position, summary,
                               trade_stats, weights_backtest)


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


def test_flip_long_to_short_pays_two_sells_and_borrow_fee():
    ret = pd.Series(0.0, index=range(4))
    bt = backtest(ret, pd.Series([1.0, -1.0, -1.0, 1.0]), 0.001, 0.004, short_fee=0.0008)
    # 買進；多翻空 = 賣 1 + 融券賣 1 + 借券費；續抱；空翻多 = 回補 1 + 買 1
    assert np.allclose(bt["cost"], [0.001, 2 * 0.004 + 0.0008, 0.0, 2 * 0.001])


def test_short_position_earns_when_price_falls():
    ret = pd.Series(np.log([0.98, 1.01]))
    bt = backtest(ret, pd.Series([-1.0, -1.0]), 0.0, 0.0)
    assert np.allclose(bt["gross"], [0.02, -0.01])


def test_signal_position_band_and_hold():
    sig = pd.Series([0.002, 0.0005, -0.0005, -0.002, np.nan, 0.0])
    assert signal_position(sig).tolist() == [1, 1, -1, -1, 0, 0]
    assert signal_position(sig, band=0.001).tolist() == [1, 0, 0, -1, 0, 0]
    assert signal_position(sig, band=0.001, hold=True).tolist() == [1, 1, 1, -1, -1, -1]
    assert signal_position(sig, long_only=True).tolist() == [1, 1, 0, 0, 0, 0]


def test_breakeven_cost_is_gross_over_turnover():
    ret = pd.Series(np.log([1.01, 1.01, 0.99, 1.0]))
    bt = backtest(ret, pd.Series([1.0, 1.0, -1.0, -1.0]), 0.001, 0.001)
    s = trade_stats(bt)
    # 毛利 0.01 + 0.01 + 0.01 + 0；換手 1 + 2 = 3
    assert np.isclose(s["turnover"], 3) and np.isclose(s["breakeven_cost"], 0.01)
    assert np.isclose(s["avg_cost"], 0.001)
    assert np.isclose(s["long_ratio"], 0.5) and np.isclose(s["short_ratio"], 0.5)


def test_random_sign_keeps_exposure():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.001, 0.01, 500))
    r = random_sign_pctile(ret, pd.Series(1.0, index=ret.index), reps=50)
    assert np.isclose(r["gross_mean"], r["random_mean"])   # 永遠做多：打亂後一模一樣
    assert np.isnan(r["pctile"])


def _w(rows, cols=("A", "B")):
    return pd.DataFrame(rows, columns=list(cols), dtype=float)


def test_weights_one_asset_reproduces_its_return():
    ret = pd.DataFrame({"A": np.log([1.02, 0.97, 1.05])})
    bt = weights_backtest(ret, _w([[1.0]] * 3, ["A"]), 0.001, 0.002)
    # 只有第一天買進付成本，之後續抱不付
    assert np.allclose(bt["cost"], [0.001, 0.0, 0.0])
    assert np.isclose(bt["equity"].iloc[-1], 0.999 * 1.02 * 0.97 * 1.05)


def test_weights_at_t_earn_return_t_to_t1():
    ret = pd.DataFrame({"A": np.log([1.10, 1.10]), "B": np.log([1.0, 0.90])})
    bt = weights_backtest(ret, _w([[1, 0], [0, 1]]), 0.0, 0.0)
    # 第 1 天收盤換到 B，第 1 天那列賺的是 B 的 t→t+1 報酬
    assert np.allclose(bt["gross"], [0.10, -0.10])


def test_weights_drift_and_band():
    ret = pd.DataFrame({"A": np.log([1.10, 1.0, 1.0]), "B": [0.0, 0.0, 0.0]})
    target = _w([[0.5, 0.5]] * 3)
    loose = weights_backtest(ret, target, 0.001, 0.002, band=0.10)
    # A 漲 10% 後權重 0.55/0.45，偏離 0.05 < band，不調整，只付第一天的買進成本
    assert np.allclose(loose[["w_A", "w_B"]].iloc[1], [0.55 / 1.05, 0.5 / 1.05])
    assert np.allclose(loose["cost"], [0.001, 0.0, 0.0])
    tight = weights_backtest(ret, target, 0.001, 0.002, band=0.0)
    d = 0.55 / 1.05 - 0.5      # 第 2 天賣 A、買 B 各 d
    assert np.isclose(tight["cost"].iloc[1], d * 0.001 + d * 0.002)


def test_weights_nan_row_holds_and_untradable_waits():
    ret = pd.DataFrame({"A": [0.0] * 3, "B": [0.0] * 3})
    target = _w([[1, 0], [np.nan, np.nan], [0, 1]])
    tradable = pd.DataFrame({"A": [True, True, True], "B": [True, True, False]})
    bt = weights_backtest(ret, target, 0.001, 0.002, tradable=tradable)
    # 第 2 天整列缺值不調整；第 3 天 B 停牌，延後，仍持有 A
    assert bt["trade"].tolist() == [True, False, False]
    assert np.allclose(bt["w_A"], [1, 1, 1])


def test_weights_cash_earns_zero():
    ret = pd.DataFrame({"A": np.log([1.10])})
    bt = weights_backtest(ret, _w([[0.5]], ["A"]), 0.0, 0.0)
    assert np.isclose(bt["net"].iloc[0], 0.05) and np.isclose(bt["position"].iloc[0], 0.5)

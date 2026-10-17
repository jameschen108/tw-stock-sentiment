import numpy as np
import pandas as pd

from pttsent import stock_portfolio as sp

D = pd.bdate_range("2020-01-27", periods=10)       # 1 月底 5 天、2 月初 5 天


def test_month_ends_skip_last_partial_month():
    assert list(sp.month_ends(D)) == [pd.Timestamp("2020-01-31")]


def test_equal_weight_drift_and_cost():
    ret = pd.DataFrame({"A": np.log(1.10), "B": 0.0}, index=D)
    t = D[4]                                          # 1/31 收盤買進 A、B 各半
    bt = sp.backtest(ret, {t: ["A", "B"]}, buy=0.001, sell=0.004)
    assert bt.index[0] == D[5] and np.isclose(bt["cost"].iloc[0], 0.001)
    # 第一天 A +10%、B 0 -> 組合 +5%，扣買進成本
    assert np.isclose(bt["net"].iloc[0], 1.05 * 0.999 - 1)
    # 第二天權重已經漂移成 1.1/2.1、1/2.1
    assert np.isclose(bt["net"].iloc[1], (0.5 * 1.21 + 0.5) / (0.5 * 1.1 + 0.5) - 1)


def test_delisted_stock_becomes_cash_and_rebalance_sells_it():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    ret = pd.DataFrame({"A": 0.0, "B": 0.0}, index=days)
    ret.loc[ret.index > "2020-02-14", "B"] = np.nan  # B 下市：之後沒有報酬
    ends = sp.month_ends(days)
    bt = sp.backtest(ret, {ends[0]: ["A", "B"], ends[1]: ["A"]}, buy=0.0, sell=0.01)
    # 2 月底把 B（50%）賣掉、A 買到 100%：賣出 0.5 × 1%
    assert np.isclose(bt.loc[bt.index > ends[1], "cost"].iloc[0], 0.005)


def test_signals_only_use_past():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2019-01-01", periods=300)
    ret = pd.DataFrame(rng.normal(0, 0.02, (300, 3)), index=days, columns=list("ABC"))
    value = pd.DataFrame(1e8, index=days, columns=list("ABC"))
    a = sp.signals(ret, value)
    ret2 = ret.copy()
    ret2.iloc[280:] = 0.5
    b = sp.signals(ret2, value)
    for k in a:
        pd.testing.assert_frame_equal(a[k].iloc[:280], b[k].iloc[:280])

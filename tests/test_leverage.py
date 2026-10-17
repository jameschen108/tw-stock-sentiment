import numpy as np
import pandas as pd

from pttsent import leverage as lv

DAYS = pd.bdate_range("2020-01-01", periods=300)


def test_c1_rebalances_on_first_trading_day_of_month():
    w = lv.c1(DAYS)
    rows = w.dropna().index
    assert rows[0] == DAYS[0]
    assert all(d.month != p.month for d, p in zip(rows[1:], rows[1:] - pd.offsets.BDay(1)))
    assert np.allclose(w.dropna().to_numpy(), 0.5)


def test_c2_leverage_from_volatility():
    rng = np.random.default_rng(0)
    calm = pd.Series(rng.normal(0, 0.05 / np.sqrt(252), 300), index=DAYS)   # 年化 5%：L 頂到 2
    wild = pd.Series(rng.normal(0, 0.60 / np.sqrt(252), 300), index=DAYS)   # 年化 60%：L 壓到 1
    assert np.allclose(lv.c2_leverage(calm).iloc[lv.VOL_WINDOW:], 2.0)
    assert np.allclose(lv.c2_leverage(wild).iloc[lv.VOL_WINDOW:], 1.0)
    assert (lv.c2_leverage(calm).iloc[: lv.VOL_WINDOW - 1] == 1.0).all()   # 不夠 60 天
    w = lv.c2(calm, DAYS).dropna()
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w.index[1:].dayofweek == 4).all()                              # 只在週五（每週最後一天）


def test_strategies_do_not_look_ahead():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0.0005, 0.01, 300), index=DAYS)
    s2 = s.copy()
    s2.iloc[200:] = -0.05                                                  # 只改未來
    for name in ("C2", "C3"):
        a, b = lv.strategies(s, DAYS)[name], lv.strategies(s2, DAYS)[name]
        pd.testing.assert_frame_equal(a.iloc[:200], b.iloc[:200])


def test_c3_holds_2x_above_moving_average():
    up = pd.Series(0.001, index=DAYS)
    w = lv.c3(up, DAYS)
    assert (w["x2"].iloc[: lv.MA_WINDOW - 1] == 0).all()                   # 均線算不出來：持有 x1
    assert (w["x2"].iloc[lv.MA_WINDOW:] == 1).all()


def test_bootstrap_win_extremes():
    assert lv.bootstrap_win(pd.Series(np.full(500, 0.001)), horizon=100, reps=200) == 1.0
    assert lv.bootstrap_win(pd.Series(np.full(500, -0.001)), horizon=100, reps=200) == 0.0


def test_holm_steps_down_and_stops():
    assert lv.holm({"C1": 0.02, "C3": 0.04}) == {"C1": True, "C3": True}     # 0.02 < 0.025、0.04 < 0.05
    assert lv.holm({"C1": 0.03, "C3": 0.04}) == {"C1": False, "C3": False}   # 第一個就沒過，後面都不過
    assert lv.holm({"C1": 0.001, "C3": 0.06}) == {"C1": True, "C3": False}

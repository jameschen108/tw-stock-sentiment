import numpy as np
import pandas as pd

from pttsent import leverage as lv
from pttsent import leverage_explore as lx

DAYS = pd.bdate_range("2020-01-01", periods=400)
SIG = pd.Series(np.random.default_rng(2).normal(0.0004, 0.012, 400), index=DAYS)


def test_registered_rules_are_special_cases():
    v = lx.variants()
    pd.testing.assert_frame_equal(lx.fixed(SIG, DAYS, 1.5), lv.c1(DAYS), check_dtype=False)
    pd.testing.assert_frame_equal(lx.trend(SIG, DAYS, 200, 1.0), lv.c3(SIG, DAYS), check_dtype=False)
    pd.testing.assert_frame_equal(lx.voltarget(SIG, DAYS, 60, 1.5), lv.c2(SIG, DAYS), check_dtype=False)
    assert v["波動60_1.5"][2] == lv.BAND_C2 and len(v) == 25


def test_trend_vol_is_1x_below_average():
    w = lx.trend_vol(SIG, DAYS, 150, 20, 1.5)
    up = lx.above_ma(SIG, 150).reindex(DAYS)
    assert np.allclose(w.loc[~up, "x2"], 0.0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_walk_forward_only_uses_past_years():
    days = pd.bdate_range("2010-01-01", "2013-12-31")
    bench = pd.Series(0.0, index=days)
    good_early = pd.Series(np.where(days.year < 2012, 0.002, -0.002), index=days)
    nets = {"ref": pd.DataFrame({"net": 0.0, "exposure": 1.5}, index=days),
            "x": pd.DataFrame({"net": good_early, "exposure": 1.5}, index=days)}
    nets["ref"].loc[days[5], "net"] = -0.5        # ref 有很深的回檔，x 一定符合回檔條件
    ret, picks = lx.walk_forward(nets, bench, 2012, "ref", switch_cost=0.0)
    # 2012 年初只看得到 2010–2011（x 很好），所以 2012 挑 x；2013 年初看到 2012 的虧損也一樣只用過去
    assert picks[2012] == "x"
    assert np.isclose(ret.loc["2012"].iloc[0], -0.002)


def test_rebalance_quarterly_and_risk_parity():
    w = lx.rebalance({"x2": 0.7, "bond": 0.3}, DAYS, "Q").dropna()
    assert all(d.month in (1, 4, 7, 10) for d in w.index[1:]) and np.allclose(w.sum(axis=1), 1.0)
    rng = np.random.default_rng(3)
    ret = pd.DataFrame({"x2": rng.normal(0, 0.02, 400), "bond": rng.normal(0, 0.01, 400)}, index=DAYS)
    rp = lx.risk_parity(ret, DAYS, ["x2", "bond"]).dropna().iloc[-1]
    assert 0.25 < rp["x2"] < 0.42 and np.isclose(rp.sum(), 1.0)      # 波動一半的資產拿約 2/3


def test_walk_forward_charges_turnover_with_weight_columns():
    days = pd.bdate_range("2010-01-01", "2012-12-31")
    bench = pd.Series(0.0, index=days)
    ref = pd.DataFrame({"net": 0.0, "w_x2": 0.5, "w_bond": 0.5}, index=days)
    ref.loc[days[3], "net"] = -0.3                                     # ref 回檔很深，其他都符合條件
    b = pd.DataFrame({"net": np.where(days.year == 2010, 0.002, 0.0), "w_x2": 1.0, "w_bond": 0.0}, index=days)
    c = pd.DataFrame({"net": np.where(days.year == 2011, 0.004, 0.0), "w_x2": 0.0, "w_bond": 1.0}, index=days)
    ret, picks = lx.walk_forward({"ref": ref, "b": b, "c": c}, bench, 2011, "ref", switch_cost=0.01)
    assert picks[2011] == "b" and picks[2012] == "c"                  # 每年只看之前的年份
    assert np.isclose(ret.loc["2011"].iloc[0], 0.0)                   # 第一年不付換手成本
    assert np.isclose(ret.loc["2012"].iloc[0], -0.01)                 # b -> c 換手 100%

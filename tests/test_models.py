import numpy as np
import pandas as pd

from pttsent.backtest import backtest
from pttsent.models import direction, hist_mean, walk_forward


def test_walk_forward_only_trains_on_the_past():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=120)
    df = pd.DataFrame({"x": rng.normal(size=120), "up_next": rng.integers(0, 2, 120).astype(float)},
                      index=idx)
    wf = walk_forward(df, ["x"], "logit", min_train=50, refit_every=10)
    assert wf.index.min() == idx[50]
    assert (wf["train_end"] < wf.index).all()


def test_backtest_charges_costs_on_position_changes():
    ret = pd.Series(0.0, index=range(4))
    bt = backtest(ret, pd.Series([1, 1, 0, 1]), buy_cost=0.001, sell_cost=0.004)
    assert np.isclose(bt["cost"].sum(), 0.001 + 0.004 + 0.001)
    assert np.isclose(bt["equity"].iloc[-1], (1 - 0.001) * (1 - 0.004) * (1 - 0.001))


def test_ridge_walk_forward_only_trains_on_the_past():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=120)
    x = rng.normal(size=120)
    df = pd.DataFrame({"x": x, "ret_next": 0.01 * x + rng.normal(0, 0.001, 120)}, index=idx)
    wf = walk_forward(df, ["x"], "ridge", target="ret_next", min_train=50, refit_every=10)
    assert wf.index.min() == idx[50] and (wf["train_end"] < wf.index).all()
    assert np.corrcoef(wf["pred"], df.loc[wf.index, "ret_next"])[0, 1] > 0.9


def test_ridge_training_does_not_see_future_targets():
    idx = pd.bdate_range("2020-01-01", periods=80)
    df = pd.DataFrame({"x": np.arange(80.0), "ret_next": 0.0}, index=idx)
    df.loc[idx[60:], "ret_next"] = 1.0                  # 只有未來的目標不是 0
    wf = walk_forward(df, ["x"], "ridge", target="ret_next", min_train=60, refit_every=20)
    assert np.allclose(wf["pred"], 0.0)


def test_direction_drops_flat_days():
    r = pd.Series([0.01, 0.0, -0.02, np.nan])
    d = direction(r)
    assert d.iloc[0] == 1.0 and d.iloc[2] == 0.0
    assert np.isnan(d.iloc[1]) and np.isnan(d.iloc[3])


def test_hist_mean_uses_only_realized_returns():
    r = pd.Series([1.0, 2.0, 3.0, 100.0])
    assert np.allclose(hist_mean(r, min_periods=1).iloc[1:], [1.0, 1.5, 2.0])

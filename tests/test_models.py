import numpy as np
import pandas as pd

from pttsent.backtest import backtest
from pttsent.models import walk_forward


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

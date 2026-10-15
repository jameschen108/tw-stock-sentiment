import numpy as np
import pandas as pd

from pttsent import etf

D = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])


def _panel(code, closes, bases, dates=D):
    return pd.DataFrame({"date": dates, "code": code, "close": closes, "base": bases})


def test_big_move_is_kept():
    # 2 倍 ETF 一天跌 20%：面板會設成缺值，這裡要保留
    p = _panel("00631L", [100, 80, 80, 80, 80], [np.nan, 100, 80, 80, 80])
    r = etf.etf_returns(p, ["00631L"], etf.load_splits())
    assert np.isclose(r.loc[D[1], "00631L"], np.log(0.8))


def test_split_divides_base():
    splits = pd.DataFrame({"code": ["X"], "date": [D[3]], "ratio": [4], "last_trade": [D[1]]})
    # 停牌 D[2]，D[3] 恢復買賣：基準 200 / 4 = 50，收 51 -> log(51/50)
    dates = D[[0, 1, 3, 4]]
    p = _panel("X", [190, 200, 51, 51], [np.nan, 190, 200, 51], dates)
    r = etf.etf_returns(p, ["X"], splits)
    assert np.isclose(r.loc[D[3], "X"], np.log(51 / 50))
    assert np.isclose(r.loc[D[4], "X"], 0.0)


def test_check_covers_suspended_days():
    idx = pd.Series([0.0, 0.01, 0.01, 0.01, 0.0], index=D)
    # ETF 在 D[2] 停牌，D[3] 的報酬涵蓋 D[2]、D[3] 兩天：2 倍 × 0.02 = 0.04，沒有誤差
    ret = pd.Series([np.nan, 0.02, 0.04, 0.0], index=D[[0, 1, 3, 4]])
    assert etf.check_vs_index(ret, idx, mult=2, tol=0.001).empty
    ret.loc[D[4]] = 0.05
    bad = etf.check_vs_index(ret, idx, mult=2, tol=0.001)
    assert list(bad.index) == [D[4]] and bad["days"].iloc[0] == 1

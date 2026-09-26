import numpy as np
import pandas as pd

from pttsent import event_study as es


def _panel():
    days = pd.bdate_range("2021-01-01", periods=40)
    rows = []
    for i, d in enumerate(days):
        rows.append((d, "1111", "TWSE", 0.01, 100.0, 1e6, 0.002))      # 每天 +1%，大盤 +0.2%
        if i != 10:                                                    # 2222 第 10 天停牌
            rows.append((d, "2222", "TPEx", -0.01, 50.0, 1e6, 0.001))
    p = pd.DataFrame(rows, columns=["date", "code", "market", "ret", "close", "value", "mkt_ret"])
    mkt = pd.DataFrame({"TWSE": 0.002, "TPEx": 0.001}, index=days)
    return p, mkt, days


def test_car_windows_and_suspension():
    p, mkt, days = _panel()
    w = es.wide(p, mkt, days, ["1111", "2222"])
    assert w["ret"].iloc[10]["2222"] == 0.0                            # 停牌日記 0，不是缺值
    ev = pd.DataFrame({"code": ["1111", "2222"], "e": [25, 8], "direction": ["bullish", "bearish"]})
    out = es.abnormal_returns(ev, w, horizons=(1, 5), pre=(5,))
    assert np.isclose(out.loc[0, "car_5"], 5 * (0.01 - 0.002))
    assert np.isclose(out.loc[0, "pre_5"], 5 * (0.01 - 0.002))
    assert np.isclose(out.loc[0, "day0"], 0.01 - 0.002)
    assert np.isclose(out.loc[1, "car_5"], 4 * (-0.01) - 5 * 0.001)    # 停牌那天只扣大盤
    late = es.abnormal_returns(pd.DataFrame({"code": ["1111"], "e": [37], "direction": ["bullish"]}), w, horizons=(5,), pre=(5,))
    assert np.isnan(late.loc[0, "car_5"])                              # 超出資料範圍 -> 缺值


def test_calendar_time_spread():
    p, mkt, days = _panel()
    w = es.wide(p, mkt, days, ["1111", "2222"])
    ev = pd.DataFrame({"code": ["1111", "2222"], "e": [20, 20], "direction": ["bullish", "bearish"]})
    s = es.calendar_time(ev, w, h=3)
    assert len(s) == 3
    assert np.allclose(s, (0.01 - 0.002) - (-0.01 - 0.001))

import pandas as pd

from pttsent.calendar import assign_trade_date

DAYS = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])   # 五、一、二


def td(ts):
    return assign_trade_date(pd.Series(pd.to_datetime([ts])), DAYS, "13:30").iloc[0]


def test_before_close_is_same_day():
    assert td("2024-01-05 13:29") == pd.Timestamp("2024-01-05")


def test_at_close_goes_to_next_trading_day():
    assert td("2024-01-05 13:30") == pd.Timestamp("2024-01-08")


def test_weekend_goes_to_monday():
    assert td("2024-01-06 10:00") == pd.Timestamp("2024-01-08")


def test_early_morning_is_same_day():
    assert td("2024-01-08 00:10") == pd.Timestamp("2024-01-08")


def test_beyond_calendar_is_nat():
    assert pd.isna(td("2024-01-09 14:00"))

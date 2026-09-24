import numpy as np
import pandas as pd

from pttsent.features import build_daily, daily_sentiment, period

DAYS = pd.bdate_range("2024-01-01", periods=5)


def texts(rows):
    return pd.DataFrame(rows, columns=["time", "account", "kind", "score", "tag"]).assign(
        time=lambda d: pd.to_datetime(d["time"]))


def test_account_weighting():
    df = texts([["2024-01-01 10:00", "a", "comment", 1.0, "push"]] * 3
               + [["2024-01-01 10:00", "b", "comment", -1.0, "boo"]])
    weighted = daily_sentiment(df, DAYS, "13:30", account_weighting=True)
    flat = daily_sentiment(df, DAYS, "13:30", account_weighting=False)
    assert weighted.loc["2024-01-01", "sent_mean"] == 0.0     # 一個帳號一票
    assert flat.loc["2024-01-01", "sent_mean"] == 0.5          # 一則一票


def test_no_signal_is_nan_not_zero():
    df = texts([["2024-01-02 10:00", "a", "comment", np.nan, "arrow"]])
    s = daily_sentiment(df, DAYS, "13:30")
    assert s.loc["2024-01-02", "n_comments"] == 1
    assert np.isnan(s.loc["2024-01-02", "sent_mean"])


def _stock():
    close = pd.Series([100, 101, 99, 102, 103.0], index=DAYS)
    return pd.DataFrame({"open": close, "close": close, "volume": 1000.0,
                         "ret": np.log(close).diff(), "oc": 0.0})


def test_target_is_next_day_return():
    stock = _stock()
    d = build_daily(stock, stock[["open", "close"]], daily_sentiment(texts([]), DAYS, "13:30"))
    assert d["ret_next"].iloc[1] == stock["ret"].iloc[2]
    assert d["up_next"].iloc[1] == 0.0 and d["up_next"].iloc[2] == 1.0


def test_dev_period_excludes_rows_targeting_final_test():
    stock = _stock()
    d = build_daily(stock, stock[["open", "close"]], daily_sentiment(texts([]), DAYS, "13:30"))
    cfg = {"split": {"start": "2024-01-01", "end": "2024-12-31",
                     "final_test_start": "2024-01-04"}}
    dev = period(d, cfg, final=False)
    assert dev["target_date"].max() < pd.Timestamp("2024-01-04")
    assert dev.index.max() == pd.Timestamp("2024-01-02")

import numpy as np
import pandas as pd
import pytest

from pttsent.volume import (abnormal, calendar_flags, cw_loss, design, in_periods, load_tx_night, mean_test,
                            preopen_counts, rolling_forecast, us_overnight)

DAYS = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])   # 五、一、二


def test_preopen_window_boundaries():
    times = pd.Series(pd.to_datetime([
        "2024-01-05 13:29",   # 週五盤中：不算週一的開盤前
        "2024-01-05 13:30",   # 週五收盤後 -> 週一
        "2024-01-06 10:00",   # 週末 -> 週一
        "2024-01-08 08:59",   # 週一開盤前 -> 週一
        "2024-01-08 09:00",   # 週一開盤：已經是盤中，不算
        "2024-01-08 13:30",   # 週一收盤後 -> 週二
    ]))
    assert preopen_counts(times, DAYS).tolist() == [0, 3, 1]


def test_preopen_window_before_data_start_is_missing():
    times = pd.Series(pd.to_datetime(["2024-01-08 08:00", "2024-01-08 20:00"]))
    n = preopen_counts(times, DAYS, start="2024-01-06")
    assert n.iloc[:2].isna().all()      # 週五沒有前一天；週一的窗口從週五 13:30 開始，早於資料起點
    assert n.iloc[2] == 1


def test_abnormal_uses_only_past():
    n = pd.Series(np.arange(30, dtype=float))
    a = abnormal(n, window=20)
    assert a.iloc[:20].isna().all()
    assert np.isclose(a.iloc[20], np.log1p(20) - np.log1p(np.arange(20)).mean())
    n2 = n.copy()
    n2.iloc[25] = 1000.0
    assert np.allclose(abnormal(n2, 20).iloc[:25], a.iloc[:25], equal_nan=True)


def test_us_overnight_alignment():
    us_days = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05",      # 1/8 美股休市
                                "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"])
    adj = pd.Series([100, 101, 102, 99, 100, 103, 104], index=us_days, dtype=float)
    us = pd.DataFrame({"adj": adj, "high": adj * 1.01, "low": adj * 0.99, "volume": 1e6})
    tw = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09",
                           "2024-01-10", "2024-01-15"])                       # 1/11–1/12 台股休市
    out = us_overnight(us, tw, "adr", volume=False)
    assert out.loc["2024-01-04"].isna().all()                                 # 沒有前一個台股交易日
    assert np.isclose(out.loc["2024-01-05", "adr_abs"], np.log(101 / 100))    # 美股 1/4 -> 台股 1/5
    assert np.isclose(out.loc["2024-01-08", "adr_abs"], np.log(102 / 101))    # 美股週五 -> 台股週一
    assert out.loc["2024-01-09", "adr_none"] == 1 and out.loc["2024-01-09", "adr_abs"] == 0
    assert np.isclose(out.loc["2024-01-10", "adr_neg"], np.log(99 / 102))
    pk1 = np.log(1.01 / 0.99) ** 2 / (4 * np.log(2))
    assert np.isclose(out.loc["2024-01-15", "adr_abs"], np.log(104 / 99))     # 連假累加 1/10–1/12
    assert np.isclose(out.loc["2024-01-15", "adr_pk"], np.sqrt(3 * pk1))


def test_calendar_flags():
    days = pd.bdate_range("2024-01-01", "2024-03-08").drop(pd.to_datetime(["2024-01-17", "2024-02-29"]))
    f = calendar_flags(days)
    assert f.index[f["settle"] == 1].strftime("%m-%d").tolist() == ["01-18", "02-21"]   # 1/17 休市 -> 順延
    assert f.index[f["msci"] == 1].strftime("%m-%d").tolist() == ["02-28"]
    assert f.loc["2024-01-02", "dow_1"] == 1
    assert np.isclose(f.loc["2024-01-08", "log_gap"], np.log(3))


def test_in_periods_includes_both_ends():
    days = pd.bdate_range("2021-06-14", periods=10)
    periods = pd.DataFrame({"start": [days[2]], "end": [days[4]]})
    assert in_periods(days, periods).tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    assert in_periods(days, periods.iloc[:0]).sum() == 0


def test_design_limit_flags_use_previous_day():
    days = pd.bdate_range("2024-01-01", periods=4)
    px = pd.DataFrame({"volume": 1e6, "high": 11.0, "low": 10.0, "ret": 0.0, "gap": 0.0,
                       "limit_up_close": [False, True, False, False],
                       "limit_down_close": [False, False, False, True]}, index=days)
    df = design(px, days, pd.Series(0.0, index=days))
    assert df["lock_up1"].tolist()[1:] == [0, 1, 0]     # 第 2 天收漲停 -> 第 3 天的特徵
    assert df["lock_dn1"].tolist()[1:] == [0, 0, 0]     # 最後一天收跌停，要到下一天才用得到


def test_tx_night_uses_previous_day_session_close(tmp_path):
    header = ("交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,漲跌價,漲跌%,成交量,結算價,未沖銷契約數,"
              "最後最佳買價,最後最佳賣價,歷史最高價,歷史最低價,是否因訊息面暫停交易,交易時段,價差對單式委託成交量")

    def row(date, month, high, low, close, vol, session):
        return f"{date},TX,{month},{close},{high},{low},{close},0,0%,{vol},{close},0,0,0,0,0,,{session},"

    lines = [header,
             row("2024/01/02", "202401  ", 101, 99, 100, 1000, "一般"),
             row("2024/01/02", "202402  ", 102, 100, 101, 10, "一般"),
             row("2024/01/03", "202401  ", 103, 99, 102, 500, "盤後"),        # 1/2 15:00 到 1/3 05:00
             row("2024/01/03", "202402  ", 105, 103, 104, 50, "盤後"),
             row("2024/01/03", "202401/202402", 1, 1, 1, 99999, "盤後"),     # 價差單不算
             row("2024/01/03", "202401  ", 106, 101, 105, 900, "一般")]
    p = tmp_path / "raw" / "taifex_tx"
    p.mkdir(parents=True)
    (p / "futDataDown_TX_202401.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = load_tx_night(tmp_path)
    t = pd.Timestamp("2024-01-03")
    assert np.isclose(out.loc[t, "n_abs"], np.log(102 / 100))    # 對 1/2 日盤收盤，不是 1/3 日盤
    assert np.isclose(out.loc[t, "n_range"], np.log(np.log(103 / 99)))


def test_rolling_forecast_uses_only_past():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    d = pd.DataFrame({"x": rng.normal(size=400)}, index=idx)
    d["y"] = 2 * d["x"] + rng.normal(size=400)
    p = rolling_forecast(d, "y", ["x"], idx[300], idx[-1], min_train=250, refit=21)
    d2 = d.copy()
    d2.loc[idx[330]:, "y"] += 100.0
    p2 = rolling_forecast(d2, "y", ["x"], idx[300], idx[-1], min_train=250, refit=21)
    # 區塊從 idx[300]、idx[321]、idx[342]… 開始；改了 idx[330] 以後的 y，前兩塊的訓練資料都在那之前
    assert np.allclose(p[:idx[341]], p2[:idx[341]])
    assert not np.allclose(p[idx[342]:], p2[idx[342]:])
    with pytest.raises(ValueError):
        rolling_forecast(d, "y", ["x"], idx[100], idx[-1], min_train=250)


def test_clark_west_detects_real_predictor():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=500))
    y = 0.5 * x + pd.Series(rng.normal(size=500))
    zero = pd.Series(np.zeros(500))
    assert mean_test(cw_loss(y, zero, 0.5 * x))["t"] > 3
    assert np.allclose(cw_loss(y, zero, zero), 0)

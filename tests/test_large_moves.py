import numpy as np
import pandas as pd

from pttsent.large_moves import (follow_on, market_model, near_event, norm_name, recheck_sheet, remove_words,
                                 superstrings)


def test_norm_name():
    assert norm_name("國巨*") == "國巨"
    assert norm_name("美食-KY") == "美食"
    assert norm_name("泰金寶-DR") == "泰金寶"
    assert norm_name("臺企銀") == "台企銀"


def test_superstrings_and_extra():
    u = pd.DataFrame({"code": ["1303"], "name": ["南亞"]})
    ex = superstrings(u, ["南亞", "南亞科", "台塑"], {"1303": ["東南亞"]})
    assert set(ex["1303"]) == {"東南亞", "南亞科"}
    text = remove_words(pd.Series(["南亞科大漲", "東南亞市場", "南亞營收"]), ex["1303"])
    assert text.str.contains("南亞").tolist() == [False, False, True]


def test_market_model_uses_only_past():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0, 0.01, 200))
    ret = 2 * mkt
    ret.iloc[170] += 0.05                     # 當天的衝擊不能影響當天的 β
    m = market_model(ret, mkt)
    assert m["beta"].iloc[:120].isna().all()  # 不足 120 天
    assert m["sigma"].iloc[159] != m["sigma"].iloc[159]   # ar 從第 120 列才有，σ 要再等 40 列
    assert np.isclose(m["beta"].iloc[170], 2)
    assert np.isclose(m["ar"].iloc[170], 0.05)
    assert m["sigma"].iloc[170] < 1e-10       # 前 60 天 ar 都是 0
    assert not np.isclose(m["beta"].iloc[171], 2)   # 隔天才進到估計窗


def test_follow_on_and_controls():
    ev = np.array([10, 13, 16, 30])
    assert follow_on(ev).tolist() == [False, True, True, False]   # 16 距前一個事件（13）3 天，也算後續
    pos = np.arange(0, 40)
    near = near_event(pos, ev)
    assert not near[4] and near[5] and near[21] and not near[22] and near[35] and not near[36]
    assert not near_event(pos, np.array([], dtype=int)).any()


def test_recheck_takes_all_when_few_and_carries_seen():
    u = pd.DataFrame({"code": ["1515"], "name": ["力山"]})
    titles = pd.DataFrame({"time": pd.to_datetime(["2020-01-01"] * 3),
                           "title": ["力山營收", "壓力山大", "力山 1515 多"]})
    seen = pd.DataFrame({"code": ["1515"], "title": ["力山營收"]})
    s = recheck_sheet(u, {"1515": ["壓力山大"]}, titles, seen, ["1515"])
    assert s["title"].tolist() == ["力山營收"]            # 有代號的不算；補了排除詞的不算
    assert s["from_round1"].tolist() == [1]


def test_day_parts_and_windows():
    from pttsent.large_moves import daily_counts, day_parts, window_counts
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=40))
    times = pd.Series(pd.to_datetime([
        "2024-02-15 08:59",   # 開盤前 -> 2/15 的 preopen
        "2024-02-15 09:00",   # 盤中 -> 2/15 的 intraday
        "2024-02-14 13:30",   # 前一天收盤後 -> 2/15 的 preopen
        "2024-02-15 13:31",   # 收盤後 -> 2/16，也就是 2/15 事件的 next
        "2024-02-08 10:00",   # 事前一週內（t−5）
    ]))
    parts = day_parts(times, days)
    pre, intra = daily_counts(parts, len(days))
    i = np.array([days.get_loc(pd.Timestamp("2024-02-15"))])
    w = window_counts(pre, intra, i).iloc[0]
    assert (w["preopen"], w["intraday"], w["next"], w["pre5"], w["post"]) == (2, 1, 1, 1, 0)
    assert w["base"] == 0
    early = window_counts(pre, intra, np.array([3])).iloc[0]
    assert np.isnan(early["pre5"]) and np.isnan(early["base"])


def test_fe_ols_matches_dummies():
    import statsmodels.api as sm
    from pttsent.large_moves import fe_ols
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"code": rng.integers(0, 8, n), "date": rng.integers(0, 15, n), "x": rng.normal(size=n)})
    df["y"] = 0.5 * df["x"] + df["code"] * 0.3 - df["date"] * 0.1 + rng.normal(size=n)
    f = fe_ols(df, "y", ["x"], ["code", "date"])
    X = pd.get_dummies(df[["code", "date"]].astype(str), drop_first=True).astype(float).assign(x=df["x"], c=1.0)
    g = sm.OLS(df["y"], X).fit()
    assert np.isclose(f.params["x"], g.params["x"])


def test_mentions_rules():
    from collections import namedtuple
    from pttsent.large_moves import mentions
    R = namedtuple("R", "code name ptt_name ptt_code exclude")
    text = pd.Series(["2008年金融海嘯", "高興昌 2008 營收", "東南亞布局", "南亞營收", "gogolook 上市"])
    assert mentions(text, R("2008", "高興昌", True, False, "")).tolist() == [False, True, False, False, False]
    assert mentions(text, R("1303", "南亞", True, True, "東南亞/南亞科")).tolist() == [False, False, False, True, False]
    assert mentions(text, R("6902", "GOGOLOOK", True, True, "")).tolist() == [False, False, False, False, True]


def test_mops_parse_and_status():
    from pttsent.mops import parse, status
    page = ("<table class='hasBorder'><tr class='tblHead'><th>公司代號</th></tr>"
            "<tr class='even'><td>&nbsp;1616</td><td>&nbsp;億泰</td><td>&nbsp;108/03/26</td><td>&nbsp;16:36:51</td>"
            "<td><pre><font size='3'>&nbsp;董事會決議召開股東會</font></pre></td>"
            "<td><input onclick=\"document.f.seq_no.value='1';\"></td></tr></table>")
    d = parse(page)
    assert status(page) == "ok" and status("資料庫中查無需求資料") == "empty" and status("<html>") == "blocked"
    assert d.iloc[0].tolist() == ["1616", pd.Timestamp("2019-03-26 16:36:51"), 1, "董事會決議召開股東會"]


def test_preopen_split():
    from pttsent.large_moves import preopen_split
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10))
    items = pd.DataFrame({"code": ["A"] * 4, "time": pd.to_datetime([
        "2024-01-04 14:00",   # 前一天收盤後 -> 1/5 early
        "2024-01-05 08:29",   # early
        "2024-01-05 08:30",   # 試撮 -> late
        "2024-01-05 09:00",   # 盤中，不算
    ])})
    d = pd.DataFrame({"code": ["A", "B"], "pos": [days.get_loc(pd.Timestamp("2024-01-05"))] * 2})
    s = preopen_split(d, items, days)
    assert s.values.tolist() == [[2, 1], [0, 0]]

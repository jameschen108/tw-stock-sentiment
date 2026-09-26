import pandas as pd

from pttsent import events


def _resolver():
    names = pd.DataFrame({
        "code": ["2603", "2618", "2023", "2330", "5871", "0050"],
        "name": ["長榮", "長榮航", "燁輝", "台積電", "中租-KY", "元大台灣50"],
        "first": pd.Timestamp("2015-01-01"), "last": pd.Timestamp("2025-01-01")})
    return events.Resolver(names, {})


def test_resolve_code_and_name():
    r = _resolver()
    d = pd.Timestamp("2021-06-01")
    assert r.resolve("[標的] 2330 台積電 多", d) == ("2330", "code+name")
    assert r.resolve("[標的] 2330.TW 多", d) == ("2330", "code")
    assert r.resolve("[標的] 台積電 空", d) == ("2330", "name")


def test_longer_name_wins_and_multi_is_excluded():
    r = _resolver()
    d = pd.Timestamp("2021-06-01")
    assert r.resolve("[標的] 長榮航 多", d)[0] == "2618"          # 不會同時算成長榮
    assert r.resolve("[標的] 長榮 長榮航 雙多", d) == (None, "multi")


def test_year_like_code_needs_its_name():
    r = _resolver()
    d = pd.Timestamp("2023-01-10")
    assert r.resolve("[標的] 2023 台積電展望 多", d) == ("2330", "name")   # 2023 是年份，不是燁輝
    assert r.resolve("[標的] 2023 燁輝 多", d)[0] == "2023"


def test_ky_name_without_suffix_and_index_posts():
    r = _resolver()
    d = pd.Timestamp("2021-06-01")
    assert r.resolve("[標的] 中租 多", d)[0] == "5871"
    assert r.resolve("[標的] 大盤 空", d) == (None, "index")
    assert r.resolve("[標的] 阿里巴巴 BABA 多", d) == (None, "no_tw_stock")


def test_inactive_names_are_not_matched():
    names = pd.DataFrame({"code": ["9999"], "name": ["舊公司"],
                          "first": pd.Timestamp("2015-01-01"), "last": pd.Timestamp("2016-01-01")})
    r = events.Resolver(names, {})
    assert r.resolve("[標的] 舊公司 多", pd.Timestamp("2020-01-01")) == (None, "no_tw_stock")


def test_title_direction():
    assert events.title_direction("[標的] 2330 台積電 多") == "bullish"
    assert events.title_direction("[標的] 3019 亞光 (空)") == "bearish"
    assert events.title_direction("[標的] 2330 多空都有") is None
    assert events.title_direction("[標的] 2603 長榮") is None


def test_entry_is_first_close_after_post():
    days = pd.DatetimeIndex(["2021-06-01", "2021-06-02", "2021-06-04"])
    t = pd.Series(pd.to_datetime(["2021-06-01 10:00", "2021-06-01 13:30", "2021-06-02 20:00", "2021-06-05 09:00"]))
    assert list(events.entry_positions(t, days)) == [0, 1, 2, -1]   # 盤中 -> 當天；收盤後 -> 下一個交易日

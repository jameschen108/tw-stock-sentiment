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

import math

import pyarrow as pa
import pyarrow.compute as pc

from pttsent.ptt import mention_pattern, strip_quotes
from pttsent.sentiment.lexicon import score_text
from pttsent.sentiment.weak_labels import label_from_content, training_text


def test_lexicon_polarity():
    assert score_text("明天噴出 歐印") == 1
    assert score_text("要崩盤了快逃") == -1


def test_lexicon_negation_and_longest_match():
    assert score_text("不會崩啦") == 1
    assert score_text("利多出盡") == -1


def test_lexicon_solo_char_and_no_signal():
    assert score_text("空!!") == -1
    assert math.isnan(score_text("今天午餐吃什麼"))


def test_strip_quotes_removes_quotes_and_templates():
    raw = ("1. 標的：6163 華電網 (例 2330.TW 台積電)\n"
           "     ex [標的] 2330.TW 台積電 長多\n"
           "※ 引述《x》之銘言：\n: 原文\n正文")
    assert "2330" not in strip_quotes(raw)
    assert strip_quotes(raw).endswith("正文")


def test_mention_pattern_respects_digit_boundaries():
    pat = mention_pattern("2330", ["台積電"])
    arr = pa.array(["2330 今天", "代號12330", "台積電噴", "23300", "(2330)"])
    assert pc.match_substring_regex(arr, pat).to_pylist() == [True, False, True, False, True]


def test_weak_labels():
    assert label_from_content("1. 標的：2330\n2. 分類：多\n") == "bullish"
    assert label_from_content("2. 分類：空\n") == "bearish"
    assert label_from_content("2. 分類：多/空/\n") is None
    assert "分類" not in training_text("2. 分類：多\n3. 分析：看好")

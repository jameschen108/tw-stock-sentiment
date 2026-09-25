import json

import pandas as pd
import pytest

from pttsent.sentiment import classifier


def _tiny_model():
    texts = ["明天加碼 會漲", "穩了 續抱", "上車 噴出", "停損 出清", "要崩了 快逃", "放空 看跌"] * 3
    labels = ["bullish"] * 3 + ["bearish"] * 3
    return classifier.build_pipeline().set_params(tfidf__min_df=1).fit(texts, labels * 3)


def test_save_writes_meta_and_load_roundtrips(tmp_path):
    path = tmp_path / "clf.joblib"
    classifier.save(_tiny_model(), path, {"labels": "weak", "data_period": [pd.Timestamp("2019-01-01")]})
    meta = classifier.load_meta(path)
    assert meta["labels"] == "weak"
    assert "sklearn_version" in meta and "trained_at" in meta
    assert classifier.score(classifier.load(path), ["明天加碼"])[0] > 0


def test_load_warns_on_sklearn_version_mismatch(tmp_path):
    path = tmp_path / "clf.joblib"
    classifier.save(_tiny_model(), path, {"labels": "weak"})
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    path.with_suffix(".json").write_text(json.dumps({**meta, "sklearn_version": "0.0.1"}), encoding="utf-8")
    with pytest.warns(UserWarning, match="0.0.1"):
        classifier.load(path)

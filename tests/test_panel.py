import json

import numpy as np
import pandas as pd

from pttsent import panel

TWSE_FIELDS = ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價"]
TPEX_FIELDS = ["代號", "名稱", "收盤", "漲跌", "開盤", "最高", "最低", "均價", "成交股數", "成交金額(元)",
               "成交筆數", "最後買價", "最後賣價", "發行股數", "次日 參考價", "次日 漲停價", "次日 跌停價"]


def _twse_day(d, rows, taiex_tr):
    return {"tables": [
        {"fields": ["報酬指數", "收盤指數"], "data": [["發行量加權股價報酬指數", str(taiex_tr)]]},
        {"fields": ["指數", "收盤指數"], "data": [["發行量加權股價指數", str(taiex_tr)]]},
        {"fields": TWSE_FIELDS, "data": [[c, n, "1,000", "10", "1,000", o, o, c_, c_] for c, n, o, c_ in rows]},
    ]}


def _tpex_day(rows):
    return {"tables": [{"fields": TPEX_FIELDS, "data": [
        [c, n, cl, "0", o, o, cl, cl, "1,000", "1,000", "1", cl, cl, "1,000,000", ref, up, "1"]
        for c, n, o, cl, ref, up in rows]}]}


def _setup(tmp_path, monkeypatch, exrights):
    work, data = tmp_path / "work", tmp_path / "data"
    (work / "prices" / "twse").mkdir(parents=True)
    (work / "prices" / "tpex").mkdir(parents=True)
    (data / "raw" / "twse_exrights").mkdir(parents=True)
    (data / "raw" / "twse_reduction").mkdir(parents=True)
    days = ["20210104", "20210105", "20210106"]
    # 上市 1111：第二天除息，參考價 95（前一天收 100），當天收 95 -> 還原後報酬應為 0
    twse = [[("1111", "甲", "100", "100")], [("1111", "甲", "95", "95")], [("1111", "甲", "95", "104.5")]]
    # 上櫃 2222：第一天的次日參考價是 45（除息），第二天收 45 -> 報酬 0；第三天開盤就漲停
    tpex = [[("2222", "乙", "50", "50", "45", "49.5")], [("2222", "乙", "45", "45", "45", "49.5")],
            [("2222", "乙", "49.5", "49.5", "49.5", "54.45")]]
    for d, tw, tp, idx in zip(days, twse, tpex, [100, 101, 102]):
        (work / "prices" / "twse" / f"{d}.json").write_text(json.dumps(_twse_day(d, tw, idx)), encoding="utf-8")
        (work / "prices" / "tpex" / f"{d}.json").write_text(json.dumps(_tpex_day(tp)), encoding="utf-8")
    (data / "raw" / "twse_exrights" / "twt49u_x.json").write_text(json.dumps(exrights), encoding="utf-8")
    return work, data


def test_exdividend_days_use_reference_price(tmp_path, monkeypatch):
    work, data = _setup(tmp_path, monkeypatch, [["110年01月05日", "1111", "甲", "100", "95"]])
    df, mkt = panel.build(work, data)
    tw = df[df["code"] == "1111"].set_index("date")
    assert np.isclose(tw.loc["2021-01-05", "ret"], 0.0)                    # 除息日沒有假跌
    assert np.isclose(tw.loc["2021-01-06", "ret"], np.log(104.5 / 95))
    tp = df[df["code"] == "2222"].set_index("date")
    assert np.isclose(tp.loc["2021-01-05", "ret"], 0.0)                    # 上櫃用次日參考價
    assert bool(tp.loc["2021-01-06", "limit_up_open"])
    assert not bool(tp.loc["2021-01-05", "limit_up_open"])
    assert np.isclose(mkt.loc["2021-01-05", "TWSE"], np.log(101 / 100))


def test_without_exrights_record_the_drop_is_kept(tmp_path, monkeypatch):
    work, data = _setup(tmp_path, monkeypatch, [])
    df, _ = panel.build(work, data)
    tw = df[df["code"] == "1111"].set_index("date")
    assert np.isclose(tw.loc["2021-01-05", "ret"], np.log(95 / 100))


def test_common_stock_filter():
    assert panel.COMMON.fullmatch("2330")
    assert not panel.COMMON.fullmatch("0050")
    assert panel.KEEP.fullmatch("00653L") and not panel.KEEP.fullmatch("712345")
    assert pd.isna(panel._f("--")) and panel._f("1,234.5") == 1234.5

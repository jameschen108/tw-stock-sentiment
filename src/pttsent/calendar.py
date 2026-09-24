"""交易日曆與貼文時間對齊。"""
import numpy as np
import pandas as pd


def assign_trade_date(times: pd.Series, trading_days: pd.DatetimeIndex,
                      cutoff: str = "13:30") -> pd.Series:
    """把每則貼文對到「收盤時已經看得到它」的交易日。

    收盤前發的 -> 當天（若當天是交易日）；收盤後、週末、假日 -> 下一個交易日。
    時間皆為台北牆上時間（naive）。超出日曆範圍 -> NaT。
    """
    t = pd.to_datetime(times)
    hh, mm = map(int, cutoff.split(":"))
    day = t.dt.normalize()
    after_close = (t - day) >= pd.Timedelta(hours=hh, minutes=mm)
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")

    days = trading_days.values.astype("datetime64[ns]")
    cand = day.values.astype("datetime64[ns]")
    pos = np.searchsorted(days, cand, side="left")
    ok = (pos < len(days)) & day.notna().values
    out = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = days[pos[ok]]
    return pd.Series(out, index=times.index)

"""研究五之後的探索：槓桿 ETF 的變化與滾動挑選（不是預先登記）。

已登記的 C0–C3 在 leverage.py，這裡不動它；這裡的函式是同一套規則加上參數，參數值在
VARIANTS 一次寫死，看結果之前就定好。資產、訊號、成本都沿用 leverage.py。
2003–2026 的資料都已經看過，這裡的結果都是樣本內；只有 walk_forward() 的挑選方法接近樣本外。
"""
import numpy as np
import pandas as pd

from . import leverage as lv


def _monthly(daily: pd.DataFrame) -> pd.DataFrame:
    month = pd.Series(daily.index.to_period("M"), index=daily.index)
    return lv._first_row_filled(daily.where(month.ne(month.shift(1))), daily)


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    week = pd.Series(daily.index.to_period("W"), index=daily.index)
    keep = week.ne(week.shift(-1)) & week.shift(-1).notna()
    return lv._first_row_filled(daily.where(keep), daily)


def _mix(L: pd.Series) -> pd.DataFrame:
    """曝險 L（1 到 2 倍）-> 0050 = 2 − L、00631L = L − 1。"""
    return pd.DataFrame({"x1": 2 - L, "x2": L - 1})


def vol_leverage(signal: pd.Series, window: int, k: float) -> pd.Series:
    vol = signal.rolling(window).std() * np.sqrt(252)
    return (k * lv.DEV_VOL / vol).clip(1.0, 2.0).fillna(1.0)


def above_ma(signal: pd.Series, n: int) -> pd.Series:
    level = signal.cumsum()
    return level > np.log(np.exp(level).rolling(n).mean())


def fixed(signal, days, L):
    """固定曝險 L，每月第一個交易日調回。L = 1.5 就是 C1。"""
    return _monthly(_mix(pd.Series(L, index=days)))


def trend(signal, days, n, low):
    """n 日均線之上 2 倍，之下 low 倍；每天檢查。n = 200、low = 1 就是 C3。"""
    up = above_ma(signal, n).reindex(days).fillna(False).astype(bool)
    return _mix(pd.Series(np.where(up, 2.0, low), index=days))


def voltarget(signal, days, window, k):
    """波動目標 k × 開發期波動，window 日實現波動；每週檢查。window = 60、k = 1.5 就是 C2。"""
    return _weekly(_mix(vol_leverage(signal, window, k).reindex(days)))


def trend_vol(signal, days, n, window, k):
    """均線之上用波動目標，之下 1 倍；每天檢查（搭配 band，只在偏離夠大時調）。"""
    up = above_ma(signal, n).reindex(days).fillna(False).astype(bool)
    L = vol_leverage(signal, window, k).reindex(days).where(up, 1.0)
    return _mix(L)


def variants() -> dict[str, tuple]:
    """名稱 -> (函式, 參數, band)。看結果之前寫死。"""
    v = {}
    for L in (1.25, 1.5, 1.75, 2.0):
        v[f"固定{L}"] = (fixed, (L,), 0.0)
    for n in (100, 150, 200, 250):
        for low in (1.0, 1.5):
            v[f"均線{n}_下{low}"] = (trend, (n, low), 0.0)
    for w in (20, 60, 120):
        for k in (1.25, 1.5, 1.75):
            v[f"波動{w}_{k}"] = (voltarget, (w, k), lv.BAND_C2)
    for n in (150, 200):
        for w in (20, 60):
            v[f"均線{n}+波動{w}"] = (trend_vol, (n, w, 1.5), lv.BAND_C2)
    return v


def rebalance(weights: dict[str, float], days, freq: str = "M") -> pd.DataFrame:
    """固定權重，每月（M）或每季（Q）第一個交易日調回。"""
    daily = pd.DataFrame(weights, index=days, dtype=float)
    per = pd.Series(days.to_period(freq), index=days)
    return lv._first_row_filled(daily.where(per.ne(per.shift(1))), daily)


def risk_parity(ret: pd.DataFrame, days, assets: list[str], window: int = 60) -> pd.DataFrame:
    """權重和過去 window 天（含當天）的波動成反比，每月第一個交易日調整；資料不夠時等權。"""
    vol = ret[assets].rolling(window).std().reindex(days)
    inv = (1 / vol).where(vol > 0)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(1 / len(assets))
    per = pd.Series(days.to_period("M"), index=days)
    return lv._first_row_filled(w.where(per.ne(per.shift(1))), w)


def bond_variants() -> dict[str, tuple]:
    """股票＋債券 ETF 的配置，名稱 -> (種類, 參數)。看結果之前寫死。x1 = 0050、x2 = 00631L、bond = 00679B。"""
    v = {}
    for eq in (50, 60, 70, 80, 90):
        v[f"正2{eq}債{100 - eq}_月"] = ("fixed", ({"x2": eq / 100, "bond": 1 - eq / 100}, "M"))
    for eq in (60, 70, 80):
        v[f"正2{eq}債{100 - eq}_季"] = ("fixed", ({"x2": eq / 100, "bond": 1 - eq / 100}, "Q"))
    for eq in (60, 80):
        v[f"0050{eq}債{100 - eq}_月"] = ("fixed", ({"x1": eq / 100, "bond": 1 - eq / 100}, "M"))
    v["0050 40正2 40債20_月"] = ("fixed", ({"x1": 0.4, "x2": 0.4, "bond": 0.2}, "M"))
    v["風險平價 正2+債_月"] = ("rp", (["x2", "bond"],))
    return v


def walk_forward(nets: dict[str, pd.DataFrame], bench: pd.Series, first_year: int, ref: str,
                 switch_cost: float) -> tuple[pd.Series, pd.Series]:
    """每年年初，只用之前所有年份的資料挑一個版本，持有一整年。

    挑選規則：最大回檔不比 ref（C1）差的版本裡，年化報酬最高的；都不符合就用 ref。
    換版本時，對換手的金額付 switch_cost（買賣各一次）：有 w_ 欄位時換手 = ½ Σ|Δw|，
    否則用 00631L 的權重（曝險 − 1）的變化。回傳（每日淨報酬, 每年挑到的版本）。
    """
    years = sorted({d.year for d in bench.index if d.year >= first_year})
    picks, parts, prev_w2 = {}, [], None
    for y in years:
        past = {k: bt.loc[: f"{y - 1}-12-31"] for k, bt in nets.items()}
        stats = {k: lv.evaluate(b["net"], bench) for k, b in past.items()}
        ok = [k for k, s in stats.items() if s["max_dd"] >= stats[ref]["max_dd"] - 1e-12]
        pick = max(ok, key=lambda k: stats[k]["cagr"]) if ok else ref
        picks[y] = pick
        cur = nets[pick].loc[str(y)]
        net = cur["net"].copy()
        wcols = [c for c in cur.columns if c.startswith("w_")]
        w = cur[wcols] if wcols else (cur[["exposure"]] - 1)
        if prev_w2 is not None and len(net):
            prev = prev_w2.reindex(w.columns, fill_value=0.0)
            extra = prev_w2.index.difference(w.columns)
            moved = (w.iloc[0] - prev).abs().sum() + prev_w2[extra].abs().sum()
            turnover = moved / 2 if wcols else moved
            net.iloc[0] = (1 + net.iloc[0]) * (1 - turnover * switch_cost) - 1
        if len(net):
            prev_w2 = w.iloc[-1]
        parts.append(net)
    return pd.concat(parts), pd.Series(picks, name="pick")

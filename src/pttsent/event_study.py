"""事件研究：發文前後的異常報酬、叢集標準誤、日曆時間組合。

事件表需要欄位：code、market、e（entry 在交易日曆上的位置，見 events.entry_positions）、direction。
報酬矩陣以交易日曆為列、股票為欄：沒成交的日子報酬記 0（下一個成交日的基準價是最後一筆，會補上漏掉的漲跌），
上市前、下市後是 NaN；所以窗口內下市或有無法還原的日子，那個事件在該期間就是缺值、不計入。

- 市場調整（主要）：個股報酬 − 所屬市場大盤報酬。
- 市場模型（穩健性）：用 e−250..e−30 估 α、β（至少 120 天）。
- 對照組合（穩健性）：entry 當天依「20 日平均成交金額」與「過去 20 日報酬」各分五等分，
  扣掉同一格所有普通股同期間的平均報酬。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def wide(panel: pd.DataFrame, market: pd.DataFrame, days: pd.DatetimeIndex, codes) -> dict:
    """回傳 ret、mkt（個股所屬市場的大盤）、value（成交金額）三個（交易日 × 股票）矩陣。
    market 是 panel.build 回傳的大盤表（欄位 TWSE、TPEx）。"""
    p = panel[panel["code"].isin(codes)]
    ret = p.pivot_table(index="date", columns="code", values="ret", aggfunc="first").reindex(days)
    traded = p.pivot_table(index="date", columns="code", values="close", aggfunc="first").reindex(days).notna()
    listed = traded.cumsum() > 0
    delisted = traded[::-1].cumsum()[::-1] == 0
    alive = listed & ~delisted
    # 有成交但報酬缺值（上市第一天、無法還原的跳動）保留 NaN；沒成交而且還在上市期間的日子記 0
    ret = ret.where(traded | ~alive, 0.0).where(alive)
    # 所屬市場逐日判斷（上櫃轉上市的股票，轉換前後扣不同的大盤）
    where = p.pivot_table(index="date", columns="code", values="market", aggfunc="first").reindex(days).ffill().bfill()
    where = where.reindex(columns=ret.columns)
    m = market.reindex(days)
    mkt = pd.DataFrame(np.where(where.to_numpy() == "TWSE", m[["TWSE"]].to_numpy(), m[["TPEx"]].to_numpy()),
                       index=days, columns=ret.columns)
    value = p.pivot_table(index="date", columns="code", values="value", aggfunc="first").reindex(days)
    return {"ret": ret, "mkt": mkt, "value": value}


def _window_sum(x: np.ndarray, rows: np.ndarray, cols: np.ndarray, start: int, end: int) -> np.ndarray:
    """每個事件 x[e+start .. e+end, col] 的總和；任何一天缺值或超出範圍 -> NaN。"""
    n = x.shape[0]
    out = np.full(len(rows), np.nan)
    for i, (e, c) in enumerate(zip(rows, cols)):
        a, b = e + start, e + end
        if a < 0 or b >= n:
            continue
        seg = x[a:b + 1, c]
        if not np.isnan(seg).any():
            out[i] = seg.sum()
    return out


def abnormal_returns(ev: pd.DataFrame, w: dict, horizons=(1, 5, 20, 60), pre=(20, 5)) -> pd.DataFrame:
    """回傳事件表加上：
    car_h / raw_h：entry 收盤之後 h 天的市場調整／原始累積報酬（主要結果）
    pre_k：發文前 k 天（到 entry 前一天收盤）的市場調整累積報酬
    day0：entry 當天（前一個收盤 -> entry 收盤，包含發文當下）的市場調整報酬
    mm_h：市場模型異常報酬；bm_h：扣掉對照組合的報酬
    """
    ret, mkt = w["ret"], w["mkt"]
    cols = pd.Index(ret.columns)
    ci = cols.get_indexer(ev["code"])
    e = ev["e"].to_numpy()
    R, M = ret.to_numpy(), mkt.to_numpy()
    AR = R - M
    out = ev.copy()
    for h in horizons:
        out[f"car_{h}"] = _window_sum(AR, e, ci, 1, h)
        out[f"raw_{h}"] = _window_sum(R, e, ci, 1, h)
    for k in pre:
        out[f"pre_{k}"] = _window_sum(AR, e, ci, -k, -1)
    out["day0"] = _window_sum(AR, e, ci, 0, 0)

    # 市場模型
    alpha, beta = np.full(len(ev), np.nan), np.full(len(ev), np.nan)
    for i, (ee, c) in enumerate(zip(e, ci)):
        a, b = ee - 250, ee - 30
        if a < 0:
            continue
        y, x = R[a:b + 1, c], M[a:b + 1, c]
        ok = ~np.isnan(y) & ~np.isnan(x)
        if ok.sum() >= 120:
            beta[i], alpha[i] = np.polyfit(x[ok], y[ok], 1)
    for h in horizons:
        out[f"mm_{h}"] = out[f"raw_{h}"] - (alpha * h + beta * _window_sum(M, e, ci, 1, h))

    # 對照組合：entry 當天的流動性 × 過去報酬五等分
    liq = w["value"].rolling(20, min_periods=10).mean()
    past = ret.rolling(20, min_periods=15).sum()
    cum = np.vstack([np.zeros((1, R.shape[1])), np.nancumsum(np.nan_to_num(R), axis=0)])
    for h in horizons:
        bm = np.full(len(ev), np.nan)
        for ee in np.unique(e):
            if ee + h >= R.shape[0]:
                continue
            fwd = cum[ee + h + 1] - cum[ee + 1]          # 每檔 e+1..e+h 的報酬總和
            valid = ~np.isnan(R[ee + 1:ee + h + 1]).any(axis=0) & liq.iloc[ee].notna().to_numpy() \
                & past.iloc[ee].notna().to_numpy()
            if valid.sum() < 50:
                continue
            q1 = pd.qcut(liq.iloc[ee][valid].rank(method="first"), 5, labels=False).to_numpy()
            q2 = pd.qcut(past.iloc[ee][valid].rank(method="first"), 5, labels=False).to_numpy()
            cell = np.full(R.shape[1], -1)
            cell[np.flatnonzero(valid)] = q1 * 5 + q2
            means = np.array([fwd[cell == k].mean() if (cell == k).any() else np.nan for k in range(25)])
            idx = np.flatnonzero(e == ee)
            cells = cell[ci[idx]]
            bm[idx] = np.where(cells >= 0, means[np.clip(cells, 0, 24)], np.nan)
        out[f"bm_{h}"] = out[f"raw_{h}"] - bm
    return out


def spread_test(df: pd.DataFrame, col: str, winsor: float | None = None) -> dict:
    """看多減看空的平均差，標準誤依 entry 日期與股票雙向叢集。"""
    d = df.dropna(subset=[col])
    y = d[col].to_numpy()
    if winsor:
        lo, hi = np.quantile(y, [winsor, 1 - winsor])
        y = np.clip(y, lo, hi)
    x = sm.add_constant((d["direction"] == "bullish").astype(float).to_numpy())
    groups = np.column_stack([pd.factorize(d["e"])[0], pd.factorize(d["code"])[0]])
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    bull, bear = y[x[:, 1] == 1], y[x[:, 1] == 0]
    return {"n_bull": len(bull), "n_bear": len(bear), "mean_bull": bull.mean(), "mean_bear": bear.mean(),
            "spread": fit.params[1], "se": fit.bse[1], "t": fit.tvalues[1],
            "p_one_sided": float(1 - norm.cdf(fit.tvalues[1])),
            "p_two_sided": float(fit.pvalues[1])}


def calendar_time(ev: pd.DataFrame, w: dict, h: int) -> pd.Series:
    """每個交易日 t：持有所有 e < t ≤ e+h 的看多事件（等權）減看空事件的市場調整報酬 -> 日報酬序列。"""
    AR = (w["ret"] - w["mkt"]).to_numpy()
    cols = pd.Index(w["ret"].columns)
    n = AR.shape[0]
    legs = {}
    for side in ("bullish", "bearish"):
        s = ev[ev["direction"] == side]
        tot, cnt = np.zeros(n), np.zeros(n)
        for ee, c in zip(s["e"], cols.get_indexer(s["code"])):
            seg = AR[ee + 1:min(ee + h + 1, n), c]
            ok = ~np.isnan(seg)
            tot[ee + 1:ee + 1 + len(seg)] += np.where(ok, seg, 0)
            cnt[ee + 1:ee + 1 + len(seg)] += ok
        legs[side] = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=w["ret"].index)
    return (legs["bullish"] - legs["bearish"]).dropna()


def newey_west_mean(x: pd.Series, lags: int) -> dict:
    fit = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"n_days": len(x), "mean_daily": fit.params[0], "t": fit.tvalues[0], "p_two_sided": float(fit.pvalues[0])}

"""研究五之後的探索：全市場個股因子組合（不是預先登記），每月換股、只做多、等權。

規則在看結果之前寫死（FACTORS 與 select()）：
- 換股日：每月最後一個交易日收盤，用當天收盤以前的資料選股，以收盤價成交，下個月生效。
- 候選：普通股（上市＋上櫃，含之後下市的），當天有成交、收盤沒有漲停或跌停（買賣不到），
  過去 252 個交易日至少 200 天有報酬，過去 60 天平均成交值排前 LIQ_TOP 名。
- 持有期間沒有報酬的日子（停牌、面板設成缺值）當作 0；下市之後的部位當作以最後價格變成現金。
成本：股票買進 0.1425%，賣出 0.1425% + 證交稅 0.3%，只對換手的金額付。
"""
import numpy as np
import pandas as pd

LIQ_TOP = 300
N_HOLD = 50
FACTORS = ["動能前50", "動能前30", "低波動前50", "成交值前50", "動能＋低波動前50"]


def month_ends(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    per = pd.Series(days.to_period("M"), index=days)
    return days[per.ne(per.shift(-1)).to_numpy() & per.shift(-1).notna().to_numpy()]


def signals(ret: pd.DataFrame, value: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """每天收盤可算的訊號（都只用當天以前的資料）。ret、value：日期 × 代號。"""
    lr = ret.fillna(0.0)
    return {
        "mom": lr.rolling(252).sum() - lr.rolling(21).sum(),          # 過去 12 個月扣掉最近 1 個月
        "vol": ret.rolling(252, min_periods=200).std(),
        "n_obs": ret.notna().rolling(252).sum(),
        "liq": value.rolling(60, min_periods=40).mean(),
    }


def select(sig: dict, t, ok_today: pd.Series, factor: str) -> list[str]:
    """t 收盤時的選股；ok_today：當天有成交、收盤沒有漲跌停的普通股。"""
    liq = sig["liq"].loc[t].where(ok_today & (sig["n_obs"].loc[t] >= 200)).dropna()
    pool = liq.nlargest(LIQ_TOP).index
    mom, vol = sig["mom"].loc[t, pool].dropna(), sig["vol"].loc[t, pool].dropna()
    if factor == "動能前50":
        return list(mom.nlargest(N_HOLD).index)
    if factor == "動能前30":
        return list(mom.nlargest(30).index)
    if factor == "低波動前50":
        return list(vol.nsmallest(N_HOLD).index)
    if factor == "成交值前50":
        return list(liq.loc[pool].nlargest(N_HOLD).index)
    if factor == "動能＋低波動前50":
        both = mom.index.intersection(vol.index)
        score = mom[both].rank(ascending=False) + vol[both].rank()
        return list(score.nsmallest(N_HOLD).index)
    raise ValueError(factor)


def backtest(ret: pd.DataFrame, picks: dict, buy: float, sell: float) -> pd.DataFrame:
    """ret：日期 × 代號的對數報酬。picks：{換股日: 代號清單}，等權。回傳每日 net、cost、n（持股數），索引是報酬發生的日子。

    兩個換股日之間權重跟著價格漂移；換股時以漂移後的權重和新目標的差付成本，記在下一個交易日。
    """
    dates = sorted(picks)
    r = np.expm1(ret.fillna(0.0))          # ret 是對數報酬
    held = pd.Series(dtype=float)
    rows = []
    for i, t in enumerate(dates):
        target = pd.Series(1.0 / len(picks[t]), index=picks[t]) if picks[t] else pd.Series(dtype=float)
        codes = held.index.union(target.index)
        diff = target.reindex(codes, fill_value=0.0) - held.reindex(codes, fill_value=0.0)
        cost = diff.clip(lower=0).sum() * buy + (-diff).clip(lower=0).sum() * sell
        end = dates[i + 1] if i + 1 < len(dates) else r.index[-1]
        period = r.loc[(r.index > t) & (r.index <= end), target.index]
        growth = (1 + period).cumprod()
        value = growth.mul(target, axis=1).sum(axis=1)
        if len(value) == 0:
            break
        daily = value / value.shift(1).fillna(1.0) - 1
        net = (1 + daily) * (1 - np.r_[cost, np.zeros(len(daily) - 1)]) - 1
        rows.append(pd.DataFrame({"net": net, "cost": np.r_[cost, np.zeros(len(daily) - 1)],
                                  "n": len(target)}, index=period.index))
        held = growth.iloc[-1] * target / value.iloc[-1]
    return pd.concat(rows)

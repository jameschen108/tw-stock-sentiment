"""簡單回測：預測會漲就在今天收盤買（或續抱），否則空手；扣台股手續費與證交稅。
target=open_to_close 時改成當沖：隔天開盤進場、收盤出場，也可以先賣後買（放空當沖）。

假設能在收盤集合競價成交。13:30 前的貼文要先算完情緒才能下單，
嚴格一點可以把 config 的 cutoff 提早（例如 13:00）。
"""
import numpy as np
import pandas as pd


def trade_costs(cfg, daytrade: bool = False):
    fee = cfg["costs"]["fee_rate"] * cfg["costs"]["fee_discount"]
    tax = cfg["costs"]["daytrade_tax_rate"] if daytrade else cfg["costs"]["tax_rate"]
    return fee, fee + tax   # (買進成本, 賣出成本)


def backtest(ret_next: pd.Series, position: pd.Series, buy_cost: float,
             sell_cost: float, round_trip: bool = False) -> pd.DataFrame:
    """position[t] ∈ {0, 1}：t 收盤到 t+1 收盤是否持有。ret_next 為對數報酬。

    round_trip=True：ret_next 是 t+1 開盤到收盤，每個有部位的日子都是一次當沖，付一次來回成本。
      position 可以是 -1（開盤先賣、收盤買回），報酬是 -(收盤/開盤 - 1)。
    假設都以開盤價、收盤價成交，沒有算滑價。
    """
    pos = position.astype(float).fillna(0.0)
    if round_trip:
        cost = pos.abs().to_numpy() * (buy_cost + sell_cost)
        trade = pos != 0
    else:
        change = pos.diff().fillna(pos.iloc[0])
        cost = np.where(change > 0, change * buy_cost, -np.minimum(change, 0) * sell_cost)
        trade = change != 0
    gross = pos * np.expm1(ret_next)
    net = gross - cost
    return pd.DataFrame({"position": pos, "gross": gross, "cost": cost, "net": net,
                         "equity": (1 + net).cumprod(), "trade": trade})


def low_sentiment_days(sent_mean: pd.Series, window: int = 250, q: float = 0.4,
                       min_periods: int = 60) -> pd.Series:
    """情緒落在過去 window 個交易日（不含當天）的最低 q 比例 -> True。

    門檻只用當天以前的資料滾動計算，不偷看未來；沒有情緒訊號的日子一律 False。
    """
    cut = sent_mean.shift(1).rolling(window, min_periods=min_periods).quantile(q)
    return (sent_mean <= cut) & sent_mean.notna()


def summary(bt: pd.DataFrame) -> dict:
    net = bt["net"]
    years = len(net) / 252
    eq = bt["equity"]
    return {
        "total_return": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": net.std() * np.sqrt(252),
        "sharpe": net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "exposure": bt["position"].abs().mean(),
        "n_trades": int(bt["trade"].sum()),
        "total_cost": bt["cost"].sum(),
    }

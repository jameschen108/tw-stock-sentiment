"""簡單回測：預測會漲就在今天收盤買（或續抱），否則空手；扣台股手續費與證交稅。

假設能在收盤集合競價成交。13:30 前的貼文要先算完情緒才能下單，
嚴格一點可以把 config 的 cutoff 提早（例如 13:00）。
"""
import numpy as np
import pandas as pd


def trade_costs(cfg):
    fee = cfg["costs"]["fee_rate"] * cfg["costs"]["fee_discount"]
    return fee, fee + cfg["costs"]["tax_rate"]   # (買進成本, 賣出成本)


def backtest(ret_next: pd.Series, position: pd.Series, buy_cost: float,
             sell_cost: float) -> pd.DataFrame:
    """position[t] ∈ {0, 1}：t 收盤到 t+1 收盤是否持有。ret_next 為對數報酬。"""
    pos = position.astype(float).fillna(0.0)
    change = pos.diff().fillna(pos.iloc[0])
    cost = np.where(change > 0, change * buy_cost, -np.minimum(change, 0) * sell_cost)
    gross = pos * np.expm1(ret_next)
    net = gross - cost
    return pd.DataFrame({"position": pos, "gross": gross, "cost": cost, "net": net,
                         "equity": (1 + net).cumprod()})


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
        "exposure": bt["position"].mean(),
        "n_trades": int((bt["position"].diff().fillna(bt["position"].iloc[0]) != 0).sum()),
        "total_cost": bt["cost"].sum(),
    }

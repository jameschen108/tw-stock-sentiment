"""簡單回測：預測會漲就在今天收盤買（或續抱），否則空手或放空；扣台股手續費與證交稅。
target=open_to_close 時改成當沖：隔天開盤進場、收盤出場，也可以先賣後買（放空當沖）。
放空有兩種成本：現股（收盤到收盤要融券，另付借券費）與個股期貨（期交稅低、多空對稱）。

假設能在收盤集合競價成交。13:30 前的貼文要先算完情緒才能下單，
嚴格一點可以把 config 的 cutoff 提早（例如 13:00）。
"""
import numpy as np
import pandas as pd


def trade_costs(cfg, daytrade: bool = False):
    fee = cfg["costs"]["fee_rate"] * cfg["costs"]["fee_discount"]
    tax = cfg["costs"]["daytrade_tax_rate"] if daytrade else cfg["costs"]["tax_rate"]
    return fee, fee + tax   # (買進成本, 賣出成本)


def cost_model(cfg, instrument: str = "stock", daytrade: bool = False):
    """回傳 (買進成本, 賣出成本, 融券借券費)，都是成交金額的比例。

    stock：現股；收盤到收盤放空要融券，賣出時另付借券費。當沖先賣後買不用融券。
    futures：個股期貨，買賣都付手續費加期交稅，沒有證交稅與借券費；以現股報酬近似期貨報酬（忽略基差）。
    """
    if instrument == "stock":
        buy, sell = trade_costs(cfg, daytrade)
        return buy, sell, 0.0 if daytrade else cfg["costs"]["short_fee_rate"]
    if instrument == "futures":
        c = cfg["costs"]["futures"]["fee_rate"] + cfg["costs"]["futures"]["tax_rate"]
        return c, c, 0.0
    raise ValueError(instrument)


def signal_position(signal: pd.Series, band: float = 0.0, long_only: bool = False,
                    hold: bool = False) -> pd.Series:
    """signal > band 做多、< -band 放空，其餘空手；hold=True 時其餘維持前一天的部位。

    signal 是預測報酬（或 機率 - 0.5）。band = 0 就是單純看正負號；缺值當作沒有訊號。
    """
    pos = pd.Series(np.nan, index=signal.index)
    pos[signal > band] = 1.0
    pos[signal < -band] = 0.0 if long_only else -1.0
    if hold:
        pos = pos.ffill()
    return pos.fillna(0.0)


def backtest(ret_next: pd.Series, position: pd.Series, buy_cost: float,
             sell_cost: float, round_trip: bool = False, short_fee: float = 0.0) -> pd.DataFrame:
    """position[t] ∈ {-1, 0, 1}：t 收盤到 t+1 收盤的部位。ret_next 為對數報酬。

    買進付 buy_cost、賣出付 sell_cost（多翻空 = 賣掉多單再融券賣出，付兩次賣出成本），
    新增的空單另付 short_fee（融券借券費）。空單每天的報酬近似為 -(t+1 收盤/t 收盤 - 1)。
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
        short = (-pos).clip(lower=0)
        new_short = short.diff().fillna(short.iloc[0]).clip(lower=0)
        cost = (change.clip(lower=0) * buy_cost + (-change).clip(lower=0) * sell_cost
                + new_short * short_fee).to_numpy()
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


def turnover(bt: pd.DataFrame, round_trip: bool = False) -> float:
    """總成交金額（以部位 1 = 本金計）。當沖每個有部位的日子買賣各一次。"""
    pos = bt["position"]
    if round_trip:
        return 2 * pos.abs().sum()
    return pos.diff().fillna(pos.iloc[0]).abs().sum()


def trade_stats(bt: pd.DataFrame, round_trip: bool = False) -> dict:
    """多空比例與成本的換算。

    breakeven_cost：平均每一塊成交金額的成本要低於多少，毛利才不會被吃光（單利近似）；
    avg_cost：實際平均每一塊成交金額付了多少成本。兩者直接比，就知道訊號付不付得起成本。
    """
    to = turnover(bt, round_trip)
    pos = bt["position"]
    return {
        "long_ratio": (pos > 0).mean(),
        "short_ratio": (pos < 0).mean(),
        "gross_total": bt["gross"].sum(),
        "turnover": to,
        "breakeven_cost": bt["gross"].sum() / to if to > 0 else np.nan,
        "avg_cost": bt["cost"].sum() / to if to > 0 else np.nan,
    }


def random_sign_pctile(ret_next: pd.Series, position: pd.Series, reps: int = 2000,
                       seed: int = 0) -> dict:
    """把部位隨機打亂（多、空、空手的天數不變），看實際的毛報酬落在隨機分布的哪裡。

    多空比例相同時，賺賠只剩「哪幾天做多」的差別，用來把偏多偏空的曝險和預測能力分開。
    只比毛報酬：打亂會破壞部位的連續性，換手變多，比淨報酬不公平。
    """
    m = pd.concat([ret_next, position], axis=1, keys=["r", "p"]).dropna()
    r, p = np.expm1(m["r"].to_numpy()), m["p"].to_numpy()
    if len(np.unique(p)) < 2:   # 部位從頭到尾一樣（例如永遠做多），打亂後沒有差別
        return {"gross_mean": (p * r).mean(), "random_mean": (p * r).mean(),
                "pctile": np.nan, "p_one_sided": np.nan}
    rng = np.random.default_rng(seed)
    sims = np.array([(rng.permutation(p) * r).mean() for _ in range(reps)])
    actual = (p * r).mean()
    return {"gross_mean": actual, "random_mean": sims.mean(),
            "pctile": (sims < actual).mean(), "p_one_sided": (sims >= actual).mean()}

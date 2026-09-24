"""每日特徵表：情緒、關注度、量價控制變數與預測目標。

每一列是交易日 t，所有特徵都只用 t 收盤前（13:30）看得到的資訊；
目標是 t+1 的報酬。
"""
import numpy as np
import pandas as pd

from .calendar import assign_trade_date

PRICE_FEATURES = ["ret", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_5",
                  "vol_20", "vlm_z", "oc", "mkt_ret"]
SENT_FEATURES = ["sent", "has_sent", "sent_chg", "bull_ratio", "sent_std",
                 "log_posts", "attn_abn", "push_ratio"]


def daily_sentiment(texts: pd.DataFrame, days: pd.DatetimeIndex, cutoff: str,
                    account_weighting: bool = True) -> pd.DataFrame:
    """texts 需要欄位：time account kind score tag。score 為 NaN 表示這則沒有情緒訊號。

    回傳每個交易日：
      sent_mean   平均情緒（帳號加權時：先在帳號內平均，再跨帳號平均）
      bull_ratio  有訊號的單位中偏多的比例
      sent_std    情緒分歧（標準差）
      n_polar     有訊號的單位數（帳號或文字）
      n_articles / n_comments / n_accounts  關注度（不論有無訊號）
      push_ratio  推 /（推 + 噓）
    沒有任何有訊號文字的交易日，情緒欄位是 NaN，不是 0。
    """
    df = texts.assign(td=assign_trade_date(texts["time"], days, cutoff)).dropna(subset=["td"])
    g = df.groupby("td")
    out = pd.DataFrame({
        "n_articles": g["kind"].apply(lambda s: (s == "article").sum()),
        "n_comments": g["kind"].apply(lambda s: (s == "comment").sum()),
        "n_accounts": g["account"].nunique(),
    })
    tags = df[df["tag"].isin(["push", "boo"])].groupby("td")["tag"]
    out["push_ratio"] = tags.apply(lambda s: (s == "push").mean())

    polar = df.dropna(subset=["score"])
    unit = (polar.groupby(["td", "account"])["score"].mean() if account_weighting
            else polar.set_index("td")["score"])
    ug = unit.groupby(level="td")
    out["sent_mean"] = ug.mean()
    out["bull_ratio"] = ug.apply(lambda s: (s > 0).mean())
    out["sent_std"] = ug.std(ddof=0)
    out["n_polar"] = ug.size()

    out = out.reindex(days).astype(float)
    for c in ("n_articles", "n_comments", "n_accounts", "n_polar"):
        out[c] = out[c].fillna(0).astype(int)
    out.index.name = "date"
    return out


def price_features(stock: pd.DataFrame, taiex: pd.DataFrame) -> pd.DataFrame:
    """用完整歷史計算（滾動視窗需要 2019 以前的資料），之後再切期間。"""
    f = pd.DataFrame(index=stock.index)
    f["ret"] = stock["ret"]
    for k in range(1, 5):
        f[f"ret_lag{k}"] = stock["ret"].shift(k)
    f["ret_5"] = stock["ret"].rolling(5).sum()
    f["vol_20"] = stock["ret"].rolling(20).std()
    lv = np.log(stock["volume"])
    f["vlm_z"] = (lv - lv.shift(1).rolling(20).mean()) / lv.shift(1).rolling(20).std()
    f["oc"] = stock["oc"]
    f["mkt_ret"] = np.log(taiex["close"]).diff().reindex(stock.index)
    f["close"] = stock["close"]
    return f


def build_daily(stock: pd.DataFrame, taiex: pd.DataFrame, sent: pd.DataFrame,
                target: str = "close_to_close") -> pd.DataFrame:
    df = price_features(stock, taiex).join(sent, how="left")

    # 情緒缺值補 0 並用 has_sent 標記，模型才吃得進去；分析時用原始的 sent_mean
    df["has_sent"] = df["sent_mean"].notna().astype(float)
    df["sent"] = df["sent_mean"].fillna(0.0)
    df["sent_chg"] = df["sent"] - df["sent"].shift(1).rolling(5).mean()
    df["bull_ratio"] = df["bull_ratio"].fillna(0.5)
    df["sent_std"] = df["sent_std"].fillna(0.0)
    df["push_ratio"] = df["push_ratio"].fillna(df["push_ratio"].median())
    df["log_posts"] = np.log1p(df["n_articles"].fillna(0) + df["n_comments"].fillna(0))
    df["attn_abn"] = df["log_posts"] - df["log_posts"].shift(1).rolling(20).mean()

    nxt = stock["ret"] if target == "close_to_close" else stock["oc"]
    df["ret_next"] = nxt.shift(-1)
    df["up_next"] = (df["ret_next"] > 0).astype(float).where(df["ret_next"].notna())
    df["excess_next"] = (stock["ret"] - df["mkt_ret"]).shift(-1)
    df["target_date"] = pd.Series(df.index, index=df.index).shift(-1)
    return df


def period(df: pd.DataFrame, cfg, final: bool = False) -> pd.DataFrame:
    """切出分析期間。沒加 --final 時，目標日落在最終測試期的列一律不出現。"""
    start = pd.Timestamp(cfg["split"]["start"])
    end = pd.Timestamp(cfg["split"]["end"])
    test_start = pd.Timestamp(cfg["split"]["final_test_start"])
    df = df[(df.index >= start) & (df["target_date"] <= end)]
    if not final:
        df = df[df["target_date"] < test_start]
    return df.dropna(subset=["target_date"])

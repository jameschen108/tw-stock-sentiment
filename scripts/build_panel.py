"""把 fetch_prices.py 下載的日檔整理成全市場面板，並驗證。

    python scripts/build_panel.py

輸出 data/prices/panel.parquet（每檔每個交易日一列）與 market.parquet（上市、上櫃大盤報酬）。
驗證：
1. 共用資料夾加權指數日曆上的每個交易日，兩個交易所都要有資料。
2. 和共用資料夾 FinMind 的 267 檔上市股比對收盤價與還原後報酬。
3. 上櫃沒有外部對照：看除權息日（次日參考價 ≠ 收盤）的報酬有沒有系統性下跌、上櫃大盤和上市大盤的相關。
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import panel  # noqa: E402
from pttsent.config import load_config, work_path  # noqa: E402
from pttsent.prices import load_stock, trading_days  # noqa: E402


def main():
    cfg = load_config()
    df, mkt = panel.build(cfg["work_dir"], cfg["data_dir"])
    out = work_path(cfg, "prices", "panel.parquet")
    df.to_parquet(out, index=False)
    mkt.to_parquet(out.with_name("market.parquet"))
    print(f"面板 {len(df):,} 列，{df['date'].min().date()} .. {df['date'].max().date()}，"
          f"{df['code'].nunique():,} 檔（普通股 {df.loc[df['common'], 'code'].nunique():,}）-> {out}")
    print(df.groupby("market")["code"].nunique().to_dict())

    print("\n1. 交易日")
    cal = trading_days(cfg["data_dir"])
    for m in ("TWSE", "TPEx"):
        have = set(df.loc[df["market"] == m, "date"])
        in_range = cal[(cal >= df["date"].min()) & (cal <= df["date"].max())]
        miss = [d.date().isoformat() for d in in_range if d not in have]
        print(f"  {m}：日曆 {len(in_range)} 天，缺 {len(miss)} 天 {miss[:10]}")
    extra = sorted(set(df["date"]) - set(cal))
    print(f"  面板有、日曆沒有的日子 {len(extra)} 天（日曆只到 {cal.max().date()}）")

    print("\n2. 和 FinMind 上市股比對")
    uni = pd.read_csv(Path(cfg["data_dir"]) / "universe_267.csv", dtype=str, encoding="utf-8-sig")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in uni["ticker"]:
            try:
                f = load_stock(cfg["data_dir"], t)
            except FileNotFoundError:
                continue
            p = df[df["code"] == t].set_index("date")
            j = p[["close", "ret"]].join(f[["close", "ret"]], rsuffix="_fm", how="inner")
            both = j.dropna(subset=["ret", "ret_fm"])
            rows.append({"code": t, "days": len(j),
                         "close_diff": int((j["close"] - j["close_fm"]).abs().gt(1e-6).sum()),
                         "ret_diff": int((both["ret"] - both["ret_fm"]).abs().gt(1e-6).sum()),
                         "ret_only_ours": int((j["ret"].notna() & j["ret_fm"].isna()).sum()),
                         "ret_only_fm": int((j["ret"].isna() & j["ret_fm"].notna()).sum())})
    r = pd.DataFrame(rows)
    print(f"  {len(r)} 檔、{r['days'].sum():,} 個檔日：收盤不同 {r['close_diff'].sum()}，"
          f"報酬不同 {r['ret_diff'].sum()}，只有我們有報酬 {r['ret_only_ours'].sum()}，"
          f"只有 FinMind 有 {r['ret_only_fm'].sum()}")
    worst = r.sort_values(["close_diff", "ret_diff"], ascending=False).head(5)
    print(worst.to_string(index=False))
    r.to_csv(out.with_name("validation_finmind.csv"), index=False)

    print("\n3. 上櫃內部檢查")
    otc = df[(df["market"] == "TPEx") & df["common"]].copy()
    prev_close = otc.groupby("code")["close"].shift(1)
    exd = otc["base"].sub(prev_close).abs() > 1e-6      # 基準價和前一日收盤不同：除權息或減資
    ar = otc["ret"] - otc["mkt_ret"]
    print(f"  除權息／減資日 {int(exd.sum()):,} 個：異常報酬平均 {ar[exd].mean():+.4f}，"
          f"其他日 {ar[~exd].mean():+.4f}（若沒還原，除息日平均會明顯為負）")
    naive = np.log(otc["close"] / prev_close)
    print(f"  同一批日子如果不還原，平均報酬 {naive[exd].mean():+.4f}，還原後 {otc.loc[exd, 'ret'].mean():+.4f}")
    print(f"  上櫃大盤 vs 上市大盤日報酬相關 {mkt['TPEx'].corr(mkt['TWSE']):.3f}")
    print(f"  報酬超過漲跌幅被設成缺值的比例：上市 {df.loc[df.market == 'TWSE', 'ret'].isna().mean():.4f}，"
          f"上櫃 {df.loc[df.market == 'TPEx', 'ret'].isna().mean():.4f}（含每檔第一天）")

    meta = {"n_rows": len(df), "period": [str(df["date"].min().date()), str(df["date"].max().date())],
            "n_codes": int(df["code"].nunique()), "finmind_close_diff": int(r["close_diff"].sum()),
            "finmind_ret_diff": int(r["ret_diff"].sum())}
    out.with_name("panel.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

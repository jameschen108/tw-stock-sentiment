"""開盤前的 PTT 討論量 → 當天成交量／振幅的預先登記檢定（見 PREREGISTRATION_VOLUME.md）。設定全部寫死。

    python scripts/11_volume_prereg.py            # 開發期（2019–2023）：參考值與檢定力
    python scripts/11_volume_prereg.py --final    # 最終測試期（2024）：預先登記 commit 後只跑一次

需要先跑：01_prepare_ptt.py --ticker 2330／2603／2317、build_panel.py、fetch_us.py。
H1（主要）：三檔 log 成交量的 Clark–West 損失差平均 > 0（Newey–West，單尾 α = 0.05）。
H2：各檔 log 成交量，Holm 校正整體 α = 0.05。
H3：三檔 log 振幅的損失差平均 > 0（單尾 α = 0.05）。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import analysis  # noqa: E402
from pttsent import volume as v  # noqa: E402
from pttsent.config import ROOT, load_config, output_path, work_path  # noqa: E402
from pttsent.prices import load_exrights  # noqa: E402

TICKERS = ["2330", "2603", "2317"]
TARGETS = {"y_lv": "log 成交量", "y_vol": "log 振幅"}
NEWS_ALIASES = {"2330": ["台積電", "台積", "TSMC"], "2603": ["長榮"], "2317": ["鴻海"]}
NEWS_EXCLUDE = {"2603": ["長榮航", "長榮空", "長榮鋼", "長榮桂冠"]}
PTT_START = "2019-01-01"                    # pttcc 的第一天；開盤前窗口早於這天的日子是缺值
DEV_END = "2023-12-31"
FINAL = ("2024-01-01", "2024-12-31")
MIN_TRAIN, REFIT, NW_LAGS, ABN_WINDOW = 250, 21, 5, 20
REGISTERED = ["PREREGISTRATION_VOLUME.md", "scripts/11_volume_prereg.py", "src/pttsent/volume.py"]


def holm(p: dict, alpha: float = 0.05) -> dict:
    """Holm 逐步校正：p 值由小到大和 alpha/(m-i) 比，一旦沒過，後面都不過。"""
    order = sorted(p, key=p.get)
    out, ok = {}, True
    for i, k in enumerate(order):
        ok = ok and p[k] < alpha / (len(p) - i)
        out[k] = ok
    return out


def committed(paths) -> bool:
    """這些檔案都已經 commit、而且之後沒有改過。"""
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def build(cfg) -> tuple:
    """每檔股票的目標與特徵（全部期間，評估時才切期間）。"""
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                            columns=["date", "code", "market", "high", "low", "volume", "ret", "gap",
                                     "limit_up_close", "limit_down_close"])
    days = pd.DatetimeIndex(sorted(panel.loc[panel["market"] == "TWSE", "date"].unique()))
    night = v.load_tx_night(cfg["data_dir"])
    us = v.us_overnight(v.load_us(work_path(cfg, "prices", "us", "TSM.json")), days, "adr").join(
        v.us_overnight(v.load_us(work_path(cfg, "prices", "us", "SOX.json")), days, "sox", volume=False)
        .drop(columns="sox_none"))                   # 和 adr_none 相同（都是美股休市）
    flags = v.calendar_flags(days)
    news = v.load_news(cfg["data_dir"])
    disp = v.load_disposition(cfg["data_dir"])
    tables = {}
    for tk in TICKERS:
        texts = pd.read_parquet(work_path(cfg, "interim", f"texts_{tk}.parquet"), columns=["time"])
        pre = v.abnormal(v.preopen_counts(texts["time"], days, start=PTT_START), ABN_WINDOW)
        hit = v.news_mentions(news, tk, NEWS_ALIASES[tk], NEWS_EXCLUDE.get(tk, ()))
        news_abn = v.abnormal(v.preopen_counts(news.loc[hit, "time"], days), ABN_WINDOW).rename("news_abn")
        events = pd.DataFrame({"disp": v.in_periods(days, disp[disp["code"] == tk]),
                               "exdiv": days.isin(load_exrights(cfg["data_dir"], tk).index).astype(float)},
                              index=days)
        px = panel[panel["code"] == tk].set_index("date").sort_index()
        tables[tk] = v.design(px, days, pre, night, us, flags, news_abn.to_frame(), events)
    return tables, days


def evaluate(df: pd.DataFrame, y: str, final: bool) -> dict:
    """完整基準 vs 基準＋開盤前討論量，兩個模型用同一個樣本、同樣的重估時點。"""
    base = v.baseline_cols(y)
    d = df[[y] + base + ["pre_abn"]].dropna()
    first, last = FINAL if final else (d.index[MIN_TRAIN], DEV_END)   # 開發期前 250 列只拿來訓練
    p0 = v.rolling_forecast(d, y, base, first, last, MIN_TRAIN, REFIT)
    p1 = v.rolling_forecast(d, y, base + ["pre_abn"], first, last, MIN_TRAIN, REFIT)
    obs = d.loc[p0.index, y]
    f = v.cw_loss(obs, p0, p1)

    # 描述：評估期內的迴歸係數（Newey–West）與 1 個標準差的效果
    ev = d.loc[p0.index]
    fit = analysis.ols_hac(ev, y, base + ["pre_abn"], maxlags=NW_LAGS)
    coef = float(fit.params["pre_abn"])

    # 解讀用：再加開盤跳空（開盤後才知道）
    g = df[[y] + base + ["gap_abs", "gap_neg", "pre_abn"]].dropna()
    gb = base + ["gap_abs", "gap_neg"]
    gfirst = first if final else max(first, g.index[MIN_TRAIN])   # 跳空有缺值時，可訓練的列會少幾天
    g0 = v.rolling_forecast(g, y, gb, gfirst, last, MIN_TRAIN, REFIT)
    g1 = v.rolling_forecast(g, y, gb + ["pre_abn"], gfirst, last, MIN_TRAIN, REFIT)
    go = g.loc[g0.index, y]
    f_gap = v.cw_loss(go, g0, g1)

    return {"f": f, "f_gap": f_gap, **v.mean_test(f, NW_LAGS),
            "start": str(p0.index.min().date()), "end": str(p0.index.max().date()),
            "mse_change": float(((obs - p1) ** 2).mean() / ((obs - p0) ** 2).mean() - 1),
            "coef": coef, "coef_t": float(fit.tvalues["pre_abn"]),
            "effect_1sd": float(np.expm1(coef * ev["pre_abn"].std())),
            "mse_change_with_gap": float(((go - g1) ** 2).mean() / ((go - g0) ** 2).mean() - 1),
            "cw_t_with_gap": v.mean_test(f_gap, NW_LAGS)["t"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.final and not committed(REGISTERED):
        sys.exit("最終測試要在預先登記 commit 之後才能跑，這些檔案有未 commit 的改動：" + "、".join(REGISTERED))
    cfg = load_config()
    tag = "final" if a.final else "dev"
    tables, days = build(cfg)

    res = {(y, tk): evaluate(tables[tk], y, a.final) for y in TARGETS for tk in TICKERS}
    fs = {k: r.pop("f") for k, r in res.items()}
    fs_gap = {k: r.pop("f_gap") for k, r in res.items()}

    def pool(f, y):
        return pd.concat([f[(y, tk)] for tk in TICKERS], axis=1, join="inner").mean(axis=1)

    pooled = {y: pool(fs, y) for y in TARGETS}
    h1, h3 = v.mean_test(pooled["y_lv"], NW_LAGS), v.mean_test(pooled["y_vol"], NW_LAGS)
    h1["pass"], h3["pass"] = h1["p_one_sided"] < 0.05, h3["p_one_sided"] < 0.05
    h2 = holm({tk: res[("y_lv", tk)]["p_one_sided"] for tk in TICKERS})
    gap = v.mean_test(pool(fs_gap, "y_lv"), NW_LAGS)   # 解讀用，不是假設

    r0 = res[("y_lv", TICKERS[0])]
    print(f"{tag}：評估 {r0['start']} .. {r0['end']}（三檔共同 {h1['n']} 天）\n")
    print(f"H1（主要）三檔平均 log 成交量：CW t = {h1['t']:+.2f}，單尾 p = {h1['p_one_sided']:.4f} "
          f"-> {'通過' if h1['pass'] else '未通過'}")
    for tk in TICKERS:
        r = res[("y_lv", tk)]
        print(f"H2 {tk} log 成交量：CW t = {r['t']:+.2f}，單尾 p = {r['p_one_sided']:.4f} "
              f"-> {'通過' if h2[tk] else '未通過'}（Holm）")
    print(f"H3 三檔平均 log 振幅：CW t = {h3['t']:+.2f}，單尾 p = {h3['p_one_sided']:.4f} "
          f"-> {'通過' if h3['pass'] else '未通過'}")
    print(f"解讀（非假設）三檔平均 log 成交量，再加開盤跳空：CW t = {gap['t']:+.2f}，單尾 p = {gap['p_one_sided']:.4f}\n")

    rows = [{"股票": tk, "目標": TARGETS[y], "天數": r["n"], "MSE 變化 %": 100 * r["mse_change"],
             "CW t": r["t"], "單尾 p": r["p_one_sided"], "係數": r["coef"], "係數 t": r["coef_t"],
             "1 SD 效果 %": 100 * r["effect_1sd"], "加跳空後 MSE 變化 %": 100 * r["mse_change_with_gap"],
             "加跳空後 CW t": r["cw_t_with_gap"]} for (y, tk), r in res.items()]
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    power = {}
    if not a.final:   # 用開發期的效果換算 2024 的檢定力（偏樂觀的上限）
        n = int(((days >= FINAL[0]) & (days <= FINAL[1])).sum())
        power = {f"{y}_{tk}": v.expected_power(fs[(y, tk)], n, maxlags=NW_LAGS) for y in TARGETS for tk in TICKERS}
        power |= {f"{y}_pooled": v.expected_power(pooled[y], n, maxlags=NW_LAGS) for y in TARGETS}
        print(f"\n2024 有 {n} 個交易日，把開發期的效果當真時的檢定力（單尾 α = 0.05）：")
        for k, p in power.items():
            print(f"  {k}: 預期 t = {p['expected_t']:.2f}，檢定力 {p['power']:.0%}")

    out = output_path(cfg, "volume", f"prereg_{tag}", "result.json")
    json.dump({"H1": h1, "H2_holm": h2, "H3": h3, "with_gap_pooled_volume": gap,
               "by_stock": {f"{y}_{tk}": r for (y, tk), r in res.items()}, "power": power},
              open(out, "w"), ensure_ascii=False, indent=2, default=float)
    pd.DataFrame({f"{y}_{tk}": f for (y, tk), f in fs.items()}).to_parquet(out.with_name("cw_loss.parquet"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

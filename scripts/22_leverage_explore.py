"""研究五之後的探索（不是預先登記）：找回檔不比 C1 差、總報酬更高的策略。

    python scripts/22_leverage_explore.py lev    # 槓桿 ETF 的 25 個變化，2003-01 .. 2026-09，加上滾動挑選
    python scripts/22_leverage_explore.py bond   # 股票＋債券 ETF（00679B）的 12 個配置，2017-01 .. 2026-09，加上滾動挑選
    python scripts/22_leverage_explore.py factor # 全市場個股因子組合 5 個，2016-01 .. 2026-09，加上滾動挑選
    python scripts/22_leverage_explore.py common # 三類的代表放在共同期間 2017-01 .. 2026-09 比

「更好」的標準：同一段期間，最大回檔不比 C1（固定 1.5 倍）差，扣成本後的年化報酬最高。
2003–2026 的資料都已經看過，全期的排名是樣本內；滾動挑選每年只用過去的資料挑版本，比較接近樣本外。
輸出在 output/leverage_explore/。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pttsent import etf  # noqa: E402
from pttsent import leverage as lv  # noqa: E402
from pttsent import leverage_explore as lx  # noqa: E402
from pttsent import stock_portfolio as sp  # noqa: E402
from pttsent.backtest import etf_costs, trade_costs, weights_backtest  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402
from pttsent.event_study import newey_west_mean  # noqa: E402

START, END = "2003-01-02", "2026-09-24"
REF = "固定1.5"                  # = C1
WF_FIRST_YEAR = 2008             # 滾動挑選至少用 5 年的資料
BOND = "00679B"                  # 元大美債 20 年，2017-01-17 上櫃；美元計價、沒有避險
BOND_START = "2017-01-17"        # 00679B 第一天收盤建倉
BOND_WF_FIRST_YEAR = 2020        # 至少用 3 年的資料


def run_all(inp, cfg) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    ret, tradable = lv.assets(inp, START, END)
    days = ret.index
    buy, sell = etf_costs(cfg)
    specs = {"C0（0050）": (lambda s, d: lv.c0(d), (), 0.0), **lx.variants()}
    out = {}
    for name, (fn, args, band) in specs.items():
        w = fn(inp["signal"], days, *args)
        bt = weights_backtest(ret.shift(-1), w, buy, sell, band=band, tradable=tradable)
        bt = bt.assign(exposure=bt["w_x1"] + 2 * bt["w_x2"])[["net", "cost", "trade", "exposure"]].iloc[:-1]
        bt.index = days[1:]
        out[name] = bt
    return out, inp["bench"].reindex(days[1:])


def table(nets, bench, start, end) -> pd.DataFrame:
    rows = {}
    for name, bt in nets.items():
        b = bt.loc[start:end]
        r = lv.evaluate(b["net"], bench)
        r.update(avg_exposure=b["exposure"].mean(), trades=int(b["trade"].sum()), cost=b["cost"].sum())
        rows[name] = r
    t = pd.DataFrame(rows).T
    t["ok_dd"] = t["max_dd"] >= t.loc[REF, "max_dd"] - 1e-12
    return t.sort_values("cagr", ascending=False)


def vs_ref(nets, name, start, end) -> dict:
    """這個版本對 C1 的每日對數超額報酬：Newey-West t。"""
    a = np.log1p(nets[name]["net"].loc[start:end])
    b = np.log1p(nets[REF]["net"].loc[start:end])
    return newey_west_mean(a - b, 20)


def cmd_lev(cfg):
    out = output_path(cfg, "leverage_explore", "x").parent
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "code", "close", "base"])
    inp = lv.load_inputs(cfg, panel)
    nets, bench = run_all(inp, cfg)
    pd.set_option("display.width", 220)
    cols = ["cagr", "excess_cagr", "ann_vol", "max_dd", "worst_12m", "win_3y", "nw_t", "avg_exposure", "trades",
            "cost", "ok_dd"]
    for label, (s, e) in {"2003-01 .. 2026-09": (START, END), "2015-01 .. 2026-09（實際 ETF）": ("2015-01-06", END)}.items():
        t = table(nets, bench, s, e)
        t.to_csv(out / f"lev_{s[:4]}.csv")
        print(f"\n== {label}：TAIEX TR 年化 {t['bench_cagr'].iloc[0]:.3f}；C1 最大回檔 {t.loc[REF, 'max_dd']:.3f}")
        print(t[cols].round(3).to_string())
        ok = t[t["ok_dd"]]
        better = ok[ok["cagr"] > t.loc[REF, "cagr"]]
        print(f"回檔不比 C1 差的有 {len(ok)} 個，其中年化報酬比 C1 高的 {len(better)} 個：{list(better.index)}")
        for name in better.index[:3]:
            r = vs_ref(nets, name, s, e)
            print(f"  {name} 對 C1 的每日超額：年化 {r['mean_daily'] * 252:+.4f}，NW t = {r['t']:.2f}")

    buy, sell = etf_costs(cfg)
    wf, picks = lx.walk_forward(nets, bench, WF_FIRST_YEAR, REF, switch_cost=buy + sell)
    picks.to_csv(out / "lev_walk_forward_picks.csv")
    s = f"{WF_FIRST_YEAR}-01-01"
    wf_stats = lv.evaluate(wf.loc[s:END], bench)
    ref_stats = lv.evaluate(nets[REF]["net"].loc[s:END], bench)
    print(f"\n== 滾動挑選 {WF_FIRST_YEAR} .. 2026-09（每年年初只用過去的資料挑版本）")
    print(picks.to_string())
    print(pd.DataFrame({"滾動挑選": wf_stats, "C1": ref_stats}).loc[
        ["cagr", "bench_cagr", "excess_cagr", "max_dd", "worst_12m", "nw_t"]].round(3).to_string())
    d = np.log1p(wf.loc[s:END]) - np.log1p(nets[REF]["net"].loc[s:END])
    r = newey_west_mean(d.dropna(), 20)
    print(f"滾動挑選對 C1 的每日超額：年化 {r['mean_daily'] * 252:+.4f}，NW t = {r['t']:.2f}")
    print(f"\n輸出在 {out}")


def run_bond(cfg, panel, bench_all) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """0050、00631L、00679B 都用實際 ETF 報酬。債券 ETF 的證交稅目前停徵，這裡仍照 0.1% 算，偏保守。"""
    codes = {"x1": "0050", "x2": "00631L", "bond": BOND}
    r = etf.etf_returns(panel, list(codes.values()), etf.load_splits()).rename(columns={v: k for k, v in codes.items()})
    days = bench_all.loc[BOND_START:END].index
    ret = r.reindex(days)
    px = panel[panel["code"].isin(codes.values())].pivot(index="date", columns="code", values="close")
    tradable = px.rename(columns={v: k for k, v in codes.items()}).reindex(days).notna()
    buy, sell = etf_costs(cfg)
    specs = {"C0（0050）": ("fixed", ({"x1": 1.0}, "M")), "固定1.5": ("fixed", ({"x1": 0.5, "x2": 0.5}, "M")),
             **lx.bond_variants()}
    out = {}
    for name, (kind, args) in specs.items():
        if kind == "fixed":
            w = lx.rebalance(args[0], days, args[1])
        else:
            w = lx.risk_parity(ret, days, *args)
        hold = w.isna().all(axis=1)                       # 整列缺值 = 這天不調整，補欄位之後也要維持
        w = w.reindex(columns=list(codes), fill_value=0.0)
        w.loc[hold] = np.nan
        bt = weights_backtest(ret.shift(-1), w, buy, sell, tradable=tradable)
        bt = bt.assign(exposure=bt["w_x1"] + 2 * bt["w_x2"])
        bt = bt[["net", "cost", "trade", "exposure", "w_x1", "w_x2", "w_bond"]].iloc[:-1]
        bt.index = days[1:]
        out[name] = bt
    return out, bench_all.reindex(days[1:])


def cmd_bond(cfg):
    out = output_path(cfg, "leverage_explore", "x").parent
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "code", "close", "base"])
    inp = lv.load_inputs(cfg, panel)
    nets, bench = run_bond(cfg, panel, inp["bench"])
    s, e = bench.index[0], END
    t = table(nets, bench, s, e)
    t["bond_w"] = [nets[k]["w_bond"].mean() for k in t.index]
    t.to_csv(out / "bond_2017.csv")
    pd.set_option("display.width", 220)
    cols = ["cagr", "excess_cagr", "ann_vol", "max_dd", "worst_12m", "win_3y", "nw_t", "avg_exposure", "bond_w",
            "trades", "cost", "ok_dd"]
    print(f"== {s.date()} .. {e}：TAIEX TR 年化 {t['bench_cagr'].iloc[0]:.3f}；C1 最大回檔 {t.loc[REF, 'max_dd']:.3f}")
    print(t[cols].round(3).to_string())
    ok = t[t["ok_dd"]]
    better = ok[ok["cagr"] > t.loc[REF, "cagr"]]
    print(f"回檔不比 C1 差的有 {len(ok)} 個，其中年化報酬比 C1 高的 {len(better)} 個：{list(better.index)}")
    for name in better.index[:3]:
        r = vs_ref(nets, name, s, e)
        print(f"  {name} 對 C1 的每日超額：年化 {r['mean_daily'] * 252:+.4f}，NW t = {r['t']:.2f}")
    print("\n壓力期間累積報酬")
    stress = {"2020-02..03": ("2020-02-20", "2020-03-19"), "2022": ("2022-01-01", "2022-12-31"),
              "2024-07..08": ("2024-07-11", "2024-08-05"), "2025-03..04": ("2025-03-01", "2025-04-09")}
    st = {lab: {k: np.expm1(np.log1p(nets[k]["net"].loc[a:b]).sum()) for k in t.index[:6].tolist() + [REF, "C0（0050）"]}
          | {"TAIEX TR": np.expm1(bench.loc[a:b].sum())} for lab, (a, b) in stress.items()}
    print(pd.DataFrame(st).T.round(3).to_string())

    buy, sell = etf_costs(cfg)
    wf, picks = lx.walk_forward(nets, bench, BOND_WF_FIRST_YEAR, REF, switch_cost=buy + sell)
    picks.to_csv(out / "bond_walk_forward_picks.csv")
    a = f"{BOND_WF_FIRST_YEAR}-01-01"
    print(f"\n== 滾動挑選 {BOND_WF_FIRST_YEAR} .. 2026-09")
    print(picks.to_string())
    print(pd.DataFrame({"滾動挑選": lv.evaluate(wf.loc[a:END], bench), "C1": lv.evaluate(nets[REF]["net"].loc[a:END], bench)})
          .loc[["cagr", "bench_cagr", "excess_cagr", "max_dd", "worst_12m", "nw_t"]].round(3).to_string())
    d = np.log1p(wf.loc[a:END]) - np.log1p(nets[REF]["net"].loc[a:END])
    r = newey_west_mean(d.dropna(), 20)
    print(f"滾動挑選對 C1 的每日超額：年化 {r['mean_daily'] * 252:+.4f}，NW t = {r['t']:.2f}")


def run_factor(cfg, panel) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    """5 個因子組合的每日淨報酬；回傳（{名稱: bt}, 第一次選股日）。"""
    stocks = panel[panel["common"]]
    ret = stocks.pivot_table(index="date", columns="code", values="ret", aggfunc="first")
    value = stocks.pivot_table(index="date", columns="code", values="value", aggfunc="first")
    ok = stocks.assign(ok=~(stocks["limit_up_close"] | stocks["limit_down_close"])).pivot_table(
        index="date", columns="code", values="ok", aggfunc="first").reindex_like(ret).fillna(False).astype(bool)
    ret = ret.loc[:END]
    sig = sp.signals(ret, value.reindex_like(ret))
    ends = [t for t in sp.month_ends(ret.index) if sig["n_obs"].loc[t].max() >= 252 and t <= pd.Timestamp(END)]
    buy, sell = trade_costs(cfg)
    nets = {}
    for f in sp.FACTORS:
        picks = {t: sp.select(sig, t, ok.loc[t], f) for t in ends}
        bt = sp.backtest(ret, picks, buy, sell)
        nets[f] = bt.assign(exposure=1.0, trade=bt["cost"] > 0)
        print(f"{f}：{len(ends)} 次換股，平均持股 {bt['n'].mean():.0f} 檔，總成本 {bt['cost'].sum():.3f}", flush=True)
    return nets, ends[0]


def load_full_panel(cfg):
    cols_p = ["date", "code", "close", "base", "ret", "value", "common", "limit_up_close", "limit_down_close"]
    return pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=cols_p)


def cmd_factor(cfg):
    out = output_path(cfg, "leverage_explore", "x").parent
    panel = load_full_panel(cfg)
    inp = lv.load_inputs(cfg, panel[["date", "code", "close", "base"]])
    nets, start = run_factor(cfg, panel)
    buy, sell = trade_costs(cfg)
    lev_nets, bench_all = run_all(inp, cfg)
    for k in ("C0（0050）", REF):
        nets[k] = lev_nets[k]
    first = nets[sp.FACTORS[0]].index[0]
    bench = inp["bench"]
    s, e = first, END
    t = table(nets, bench, s, e)
    t.to_csv(out / "factor_2016.csv")
    pd.set_option("display.width", 220)
    cols = ["cagr", "excess_cagr", "ann_vol", "max_dd", "worst_12m", "win_3y", "nw_t", "avg_exposure", "cost", "ok_dd"]
    print(f"\n== {s.date()} .. {e}（{start.date()} 收盤第一次選股）：TAIEX TR 年化 {t['bench_cagr'].iloc[0]:.3f}；"
          f"C1 最大回檔 {t.loc[REF, 'max_dd']:.3f}")
    print(t[cols].round(3).to_string())
    okd = t[t["ok_dd"]]
    better = okd[okd["cagr"] > t.loc[REF, "cagr"]]
    print(f"回檔不比 C1 差的有 {len(okd)} 個，其中年化報酬比 C1 高的 {len(better)} 個：{list(better.index)}")
    for name in sp.FACTORS:
        a = np.log1p(nets[name]["net"].loc[s:e])
        b = np.log1p(nets["C0（0050）"]["net"].loc[s:e])
        r = newey_west_mean((a - b).dropna(), 20)
        print(f"  {name} 對 0050 的每日超額：年化 {r['mean_daily'] * 252:+.4f}，NW t = {r['t']:.2f}")
    yr = pd.DataFrame({k: np.expm1(np.log1p(v["net"].loc[s:e]).groupby(lambda d: d.year).sum()) for k, v in nets.items()})
    yr["TAIEX TR"] = np.expm1(bench.loc[s:e].groupby(lambda d: d.year).sum())
    print("\n每年報酬")
    print(yr.round(3).to_string())

    # 滾動挑選：換策略當作全部換手（個股組合之間幾乎不重疊），付股票的買賣成本，偏保守
    onehot = {k: v.loc[s:e].assign(**{f"w_{k}": 1.0}) for k, v in nets.items()}
    names = list(onehot)
    for k in names:
        for j in names:
            if f"w_{j}" not in onehot[k]:
                onehot[k][f"w_{j}"] = 0.0
    wf, picks = lx.walk_forward(onehot, bench, 2019, REF, switch_cost=buy + sell)
    picks.to_csv(out / "factor_walk_forward_picks.csv")
    print("\n== 滾動挑選 2019 .. 2026-09（每年只用過去的資料挑）")
    print(picks.to_string())
    print(pd.DataFrame({"滾動挑選": lv.evaluate(wf.loc["2019":END], bench),
                        "C1": lv.evaluate(nets[REF]["net"].loc["2019":END], bench)})
          .loc[["cagr", "bench_cagr", "excess_cagr", "max_dd", "worst_12m", "nw_t"]].round(3).to_string())


COMMON = ["C0（0050）", REF, "波動60_1.5", "均線200_下1.0", "均線150_下1.0", "波動20_1.75", "固定2.0",
          "正280債20_季", "0050 40正2 40債20_月", "成交值前50", "動能前50", "低波動前50"]
NAMES = {REF: "C1：固定 1.5 倍", "波動60_1.5": "C2：波動目標 60 日", "均線200_下1.0": "C3：均線 200"}


def cmd_common(cfg):
    """三類放在共同期間比：00679B 上櫃之後，2017-01-18 .. 2026-09-24。"""
    out = output_path(cfg, "leverage_explore", "x").parent
    panel = load_full_panel(cfg)
    inp = lv.load_inputs(cfg, panel[["date", "code", "close", "base"]])
    lev_nets, _ = run_all(inp, cfg)
    bond_nets, bench = run_bond(cfg, panel[["date", "code", "close", "base"]], inp["bench"])
    fac_nets, _ = run_factor(cfg, panel)
    allnets = {**lev_nets, **bond_nets, **fac_nets}
    s, e = bench.index[0], END
    t = table({k: allnets[k] for k in COMMON}, inp["bench"], s, e).rename(index=NAMES)
    t.to_csv(out / "common_2017.csv")
    pd.set_option("display.width", 220)
    print(f"== 共同期間 {s.date()} .. {e}：TAIEX TR 年化 {t['bench_cagr'].iloc[0]:.3f}")
    print(t[["cagr", "excess_cagr", "ann_vol", "max_dd", "worst_12m", "win_3y", "nw_t", "avg_exposure", "cost",
             "ok_dd"]].round(3).to_string())
    eq = np.exp(inp["bench"].loc[s:e].cumsum())
    print(f"TAIEX TR 最大回檔 {(eq / eq.cummax() - 1).min():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lev", "bond", "factor", "common"])
    a = ap.parse_args()
    cfg = load_config()
    {"lev": cmd_lev, "bond": cmd_bond, "factor": cmd_factor, "common": cmd_common}[a.cmd](cfg)


if __name__ == "__main__":
    main()

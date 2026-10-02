"""探索三個方向：週頻、情緒極端時才進場、相對大盤的超額報酬。只用開發期 2019–2023。

    python scripts/14_explore_directions.py

設定、12 個主要檢定與 FDR 校正都事先寫在 docs/explore_directions.md，這裡照著跑。
結果在 output/explore/：primary.csv 是主要檢定，各方向的子資料夾是次要結果（預測指標、回測、隨機基準）。
"""
import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.backtest import (backtest, cost_model, random_sign_pctile, signal_position, summary,  # noqa: E402
                              trade_stats)
from pttsent.config import load_config, output_path, work_path  # noqa: E402
from pttsent.explore import (extreme_position, extreme_spread, forward_return, hedged_return,  # noqa: E402
                             sentiment_extremes, weekly)
from pttsent.features import period  # noqa: E402
from pttsent.models import (MODEL_SETS, REG_KINDS, auc_diff_ci, direction, metrics, predict_all,  # noqa: E402
                            regression_metrics)
from pttsent.volume import cw_loss, mean_test  # noqa: E402

TICKERS = ["2330", "2603", "2317"]
METHOD = "classifier_bert"
FDR = 0.10
CONTROLS = ["ret", "ret_5", "mkt_ret", "vol_20"]


def strategies(r: pd.Series, signals: dict, cfg, periods: int = 252, hedged: bool = False,
               base: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每個訊號做「多空」與「只做多」，現股與期貨兩種成本；hedged 時每次換手另付一次期貨成本（對沖那一腳）。

    signals 的值 > 0 做多、< 0 放空（已經是部位的話直接用）；base 是要比較的基準部位（預設永遠做多）。
    """
    fut = cost_model(cfg, "futures")[0]
    rows, rand = {}, {}
    for inst in ["stock", "futures"]:
        buy, sell, short_fee = cost_model(cfg, inst)
        if hedged:
            buy, sell = buy + fut, sell + fut
        pos = {"基準": base if base is not None else pd.Series(1.0, index=r.index)}
        for name, s in signals.items():
            pos[f"多空_{name}"] = signal_position(s)
            pos[f"只做多_{name}"] = signal_position(s, long_only=True)
        for sname, p in pos.items():
            bt = backtest(r, p, buy, sell, short_fee=short_fee)
            rows[(inst, sname)] = {**summary(bt, periods), **trade_stats(bt)}
    for name, s in signals.items():
        rand[name] = random_sign_pctile(r, signal_position(s))
    bts = pd.DataFrame(rows).T
    bts.index.names = ["instrument", "strategy"]
    return bts, pd.DataFrame(rand).T


def model_direction(df: pd.DataFrame, target: str, trade_ret: pd.Series, cfg, min_train: int,
                    refit: int, periods: int, hedged: bool, out) -> dict:
    """方向一、三共用：A/B × ridge、OLS、logit，回傳主要檢定（OLS 的 Clark–West），次要結果存檔。"""
    p = predict_all(df, target, min_train, refit)
    y = df.loc[p.index, target]
    reg = pd.DataFrame({m: regression_metrics(y, p[m], p["hist_mean"])
                        for m in [f"{k}_{kind}" for kind in REG_KINDS for k in MODEL_SETS] + ["hist_mean"]}).T
    clf = pd.DataFrame({m: metrics(direction(y), p[m], y) for m in ["A_logit", "B_logit"]}).T
    cw = {kind: mean_test(cw_loss(y, p[f"A_{kind}"], p[f"B_{kind}"])) for kind in REG_KINDS}
    auc, lo, hi = auc_diff_ci(direction(y), p["A_logit"], p["B_logit"])
    signals = {f"{k}_{kind}": p[f"{k}_{kind}"] - (0.5 if kind == "logit" else 0.0)
               for k in MODEL_SETS for kind in REG_KINDS + ["logit"]}
    bts, rand = strategies(trade_ret.loc[p.index], signals, cfg, periods, hedged)
    p.to_csv(out("predictions.csv"))
    reg.to_csv(out("metrics_reg.csv"))
    clf.to_csv(out("metrics_clf.csv"))
    bts.to_csv(out("backtest.csv"))
    rand.to_csv(out("random_sign.csv"))
    pd.DataFrame({"cw_ridge": cw["ridge"], "cw_ols": cw["ols"],
                  "auc_diff": {"diff": auc, "ci_low": lo, "ci_high": hi}}).to_csv(out("tests.csv"))
    return {"n": cw["ols"]["n"], "stat": cw["ols"]["t"], "p": cw["ols"]["p_one_sided"],
            "start": p.index.min().date(), "end": p.index.max().date()}


def extreme_direction(daily: pd.DataFrame, dev_index: pd.DatetimeIndex, cfg, out) -> list:
    """方向二：情緒極端日反向進場，持有 1 天與 5 天。"""
    low, high = sentiment_extremes(daily["sent_mean"])
    test_start = pd.Timestamp(cfg["split"]["final_test_start"])
    dates = pd.Series(daily.index, index=daily.index)
    r = daily.loc[dev_index, "ret_next"]
    res, bts, rand = [], {}, {}
    for h in [1, 5]:
        keep = dev_index[(dates.shift(-h).reindex(dev_index) < test_start).to_numpy()]   # t+h 也要在開發期內
        t = extreme_spread(forward_return(daily["ret"], h).loc[keep], low.loc[keep], high.loc[keep],
                           daily.loc[keep, CONTROLS], maxlags=5 if h == 1 else 10)
        res.append({"h": h, **t})
        pos = extreme_position(low, high, h).reindex(dev_index).fillna(0.0)
        b, rd = strategies(r, {f"反向_{h}天": pos}, cfg, base=pd.Series(1.0, index=dev_index))
        bts[h], rand[h] = b, rd
    pd.DataFrame(res).to_csv(out("tests.csv"), index=False)
    pd.concat(bts, names=["hold"]).to_csv(out("backtest.csv"))
    pd.concat(rand, names=["hold"]).to_csv(out("random_sign.csv"))
    return res


def main():
    cfg = load_config()
    start, test_start = cfg["split"]["start"], cfg["split"]["final_test_start"]
    primary = []
    for t in TICKERS:
        daily = pd.read_parquet(work_path(cfg, "processed", f"daily_{t}_{METHOD}.parquet"))
        dev = period(daily, cfg)
        out = lambda d, name: output_path(cfg, "explore", t, d, name)  # noqa: E731

        w = weekly(daily, start, test_start)
        r = model_direction(w, "ret_next", w["ret_next"], cfg, min_train=52, refit=4, periods=52,
                            hedged=False, out=lambda n: out("weekly", n))
        primary.append({"direction": "一、週頻", "ticker": t, "test": "CW OLS B vs A（單尾）", **r})

        for e in extreme_direction(daily, dev.index, cfg, lambda n: out("extreme", n)):
            primary.append({"direction": "二、情緒極端", "ticker": t,
                            "test": f"悲觀 − 樂觀，{e['h']} 天（雙尾）", "n": e["n"], "stat": e["t"],
                            "p": e["p"], "effect": e["diff"], "raw_effect": e["raw_diff"],
                            "n_low": e["n_low"], "n_high": e["n_high"]})

        hedged = hedged_return(dev["ret_next"], daily["mkt_ret"].shift(-1).reindex(dev.index))
        r = model_direction(dev, "excess_next", hedged, cfg, min_train=cfg["split"]["min_train_days"],
                            refit=cfg["split"]["refit_every"], periods=252, hedged=True,
                            out=lambda n: out("excess", n))
        primary.append({"direction": "三、超額報酬", "ticker": t, "test": "CW OLS B vs A（單尾）", **r})
        print(f"{t} 完成")

    prim = pd.DataFrame(primary)
    prim["p_fdr"] = multipletests(prim["p"], alpha=FDR, method="fdr_bh")[1]
    prim["pass"] = prim["p_fdr"] < FDR
    prim.to_csv(output_path(cfg, "explore", "primary.csv"), index=False)
    print(f"\n主要檢定（{len(prim)} 個，BH FDR {FDR:.0%}）")
    print(prim.round(4).to_string(index=False))
    print(f"\n結果 -> {output_path(cfg, 'explore', 'x').parent}")


if __name__ == "__main__":
    main()

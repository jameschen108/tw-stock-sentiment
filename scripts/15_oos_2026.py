"""2026-10 起的樣本外檢定（見 docs/prereg/PREREGISTRATION_2026.md）。設定全部寫死。

    python scripts/15_oos_2026.py power             # 檢定力模擬，只用已經用過的期間（開發期、2024、2025–26）
    python scripts/15_oos_2026.py volume            # 只檢查資料涵蓋（每檔可用天數），不算任何檢定
    python scripts/15_oos_2026.py volume --final    # V1–V3：目標日 2026-10-01 .. 2027-09-30，期間結束、資料補齊後只跑一次
    python scripts/15_oos_2026.py honhai            # 用面板重現研究一 2317 盤中 A_logit 的部位（開發期、2024），並檢查資料涵蓋
    python scripts/15_oos_2026.py honhai --final    # R1：目標日 2026-10-01 .. 2028-09-30，期間結束後只跑一次

V1–V3 沿用已登記的 11_volume_prereg.py（直接載入它的 build 與 evaluate），只換評估期間與美股資料夾（data/prices/us_2027/）。
R1：研究一只用量價的 logit（A 組）預測鴻海隔天盤中（開盤到收盤）漲跌，機率 > 0.5 做多、< 0.5 放空；
檢定做多日與放空日的平均盤中報酬差 > 0（OLS 對做多虛擬變數，Newey–West 5 期，單尾 α = 0.05）。
量價改由自建面板（data/prices/panel.parquet、market.parquet）計算，因為研究一用的 FinMind 日價只到 2025-03。
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pttsent.backtest import backtest, cost_model, random_sign_pctile, summary, trade_stats  # noqa: E402
from pttsent.config import load_config, output_path, work_path  # noqa: E402
from pttsent.features import PRICE_FEATURES, price_features  # noqa: E402
from pttsent.models import direction, walk_forward  # noqa: E402

_spec = importlib.util.spec_from_file_location("volume_prereg", ROOT / "scripts" / "11_volume_prereg.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)

VOLUME_WINDOW = ("2026-10-01", "2027-09-30")
HONHAI_WINDOW = ("2026-10-01", "2028-09-30")
US_DIR = "us_2027"
HONHAI, TRAIN_START, MIN_TRAIN, REFIT, NW_LAGS, MIN_SIDE = "2317", "2019-01-01", 250, 21, 5, 10
# 研究一用過的期間，只拿來重現部位與估檢定力
USED = {"開發期": ("2020-01-01", "2023-12-31", "return_dev_oc"), "2024": ("2024-01-01", "2024-12-31", "return_2024_oc")}
REGISTERED = ["docs/prereg/PREREGISTRATION_2026.md", "scripts/15_oos_2026.py", "scripts/11_volume_prereg.py",
              "src/pttsent/volume.py", "src/pttsent/features.py", "src/pttsent/models.py"]


def committed(paths) -> bool:
    """檔案都存在、已經 commit，而且之後沒有改過。"""
    if not all((ROOT / f).exists() for f in paths):
        return False
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def guard_final(window, out: Path) -> None:
    if not committed(REGISTERED):
        sys.exit("要在預先登記 commit 之後才能跑，這些檔案不存在或有未 commit 的改動：" + "、".join(REGISTERED))
    if date.today() <= date.fromisoformat(window[1]):
        sys.exit(f"評估期到 {window[1]} 才結束，現在不能跑")
    if out.exists():
        sys.exit(f"{out} 已經存在：這個檢定只跑一次")


# ---------- V1–V3：開盤前討論量 → 當天成交量／振幅 ----------
_work_path = p.work_path


def _us_work_path(cfg, *parts):
    """11 的 build 固定讀 data/prices/us/；這次改讀 us_2027/，其餘路徑不變。"""
    if parts[:2] == ("prices", "us"):
        parts = ("prices", US_DIR) + parts[2:]
    return _work_path(cfg, *parts)


def run_volume(final: bool) -> None:
    cfg = load_config()
    out = output_path(cfg, "oos_2026", "volume", "result.json")
    if final:
        guard_final(VOLUME_WINDOW, out)
    p.work_path = _us_work_path
    p.FINAL = VOLUME_WINDOW        # evaluate(final=True) 用這個期間
    tables, days = p.build(cfg)
    n_days = int(((days >= VOLUME_WINDOW[0]) & (days <= VOLUME_WINDOW[1])).sum())
    print(f"評估期間 {VOLUME_WINDOW[0]} .. {VOLUME_WINDOW[1]}：資料裡有 {n_days} 個交易日\n")
    if not final:
        for tk in p.TICKERS:
            for y in p.TARGETS:
                d = tables[tk][[y] + p.v.baseline_cols(y) + ["pre_abn"]].dropna()
                ev = d.loc[VOLUME_WINDOW[0]:VOLUME_WINDOW[1]]
                span = f"{ev.index.min().date()} .. {ev.index.max().date()}" if len(ev) else "沒有資料"
                print(f"{tk} {p.TARGETS[y]}：評估期可用 {len(ev)} 天（{span}），之前可訓練 "
                      f"{len(d[d.index < VOLUME_WINDOW[0]])} 天")
        return

    res = {(y, tk): p.evaluate(tables[tk], y, True) for y in p.TARGETS for tk in p.TICKERS}
    fs = {k: r.pop("f") for k, r in res.items()}
    fs_gap = {k: r.pop("f_gap") for k, r in res.items()}

    def pool(f, y):
        return pd.concat([f[(y, tk)] for tk in p.TICKERS], axis=1, join="inner").mean(axis=1)

    v1, v3 = p.v.mean_test(pool(fs, "y_lv"), NW_LAGS), p.v.mean_test(pool(fs, "y_vol"), NW_LAGS)
    v1["pass"], v3["pass"] = v1["p_one_sided"] < 0.05, v3["p_one_sided"] < 0.05
    v2 = p.holm({tk: res[("y_lv", tk)]["p_one_sided"] for tk in p.TICKERS})
    gap = p.v.mean_test(pool(fs_gap, "y_lv"), NW_LAGS)

    print(f"V1（主要）三檔平均 log 成交量：CW t = {v1['t']:+.2f}，單尾 p = {v1['p_one_sided']:.4f} "
          f"-> {'通過' if v1['pass'] else '未通過'}（三檔共同 {v1['n']} 天）")
    for tk in p.TICKERS:
        r = res[("y_lv", tk)]
        print(f"V2 {tk} log 成交量：CW t = {r['t']:+.2f}，單尾 p = {r['p_one_sided']:.4f} "
              f"-> {'通過' if v2[tk] else '未通過'}（Holm）")
    print(f"V3 三檔平均 log 振幅：CW t = {v3['t']:+.2f}，單尾 p = {v3['p_one_sided']:.4f} "
          f"-> {'通過' if v3['pass'] else '未通過'}")
    print(f"解讀（非假設）三檔平均 log 成交量，再加開盤跳空：CW t = {gap['t']:+.2f}，單尾 p = {gap['p_one_sided']:.4f}\n")

    rows = [{"股票": tk, "目標": p.TARGETS[y], "天數": r["n"], "MSE 變化 %": 100 * r["mse_change"],
             "CW t": r["t"], "單尾 p": r["p_one_sided"], "係數": r["coef"], "係數 t": r["coef_t"],
             "1 SD 效果 %": 100 * r["effect_1sd"], "加跳空後 MSE 變化 %": 100 * r["mse_change_with_gap"],
             "加跳空後 CW t": r["cw_t_with_gap"]} for (y, tk), r in res.items()]
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    json.dump({"period": VOLUME_WINDOW, "V1": v1, "V2_holm": v2, "V3": v3, "with_gap_pooled_volume": gap,
               "by_stock": {f"{y}_{tk}": r for (y, tk), r in res.items()}},
              open(out, "w"), ensure_ascii=False, indent=2, default=float)
    pd.DataFrame({f"{y}_{tk}": f for (y, tk), f in fs.items()}).to_parquet(out.with_name("cw_loss.parquet"))
    print(f"\n-> {out}")


# ---------- R1：鴻海隔天盤中，只用量價的 logit ----------
def honhai_table(cfg) -> pd.DataFrame:
    """研究一的量價特徵（features.price_features），改用自建面板；ret_next = 隔天的 log(收盤 / 開盤)。"""
    px = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                         columns=["date", "code", "open", "close", "volume", "ret", "oc"])
    px = px[px["code"] == HONHAI].set_index("date").sort_index()
    mkt = pd.read_parquet(work_path(cfg, "prices", "market.parquet"))["taiex"]
    f = price_features(px, pd.DataFrame({"close": np.exp(mkt.fillna(0).cumsum())}))
    f["mkt_ret"] = mkt.reindex(f.index)      # 加權指數缺值的日子（2019-04-29、04-30）保持缺值，那幾列不訓練
    f["ret_next"] = px["oc"].shift(-1)
    f["target_date"] = pd.Series(f.index, index=f.index).shift(-1)
    return f[f.index >= TRAIN_START].dropna(subset=["target_date"])


def honhai_positions(f: pd.DataFrame, first, last) -> pd.DataFrame:
    """目標日在 [first, last] 的部位與隔天盤中簡單報酬。只用目標日 <= last 的列，所以不會看到評估期之後。"""
    d = f[f["target_date"] <= pd.Timestamp(last)]
    d = d.assign(_dir=direction(d["ret_next"]))
    prob = walk_forward(d, PRICE_FEATURES, "logit", target="_dir", min_train=MIN_TRAIN, refit_every=REFIT)["prob"]
    out = pd.DataFrame({"prob": prob, "r": np.expm1(d["ret_next"].reindex(prob.index)),
                        "target_date": d["target_date"].reindex(prob.index)})
    out["pos"] = np.sign(out["prob"] - 0.5)          # 剛好 0.5 = 空手，不進檢定
    return out[out["target_date"] >= pd.Timestamp(first)].dropna(subset=["r"])


def r1_test(pos: np.ndarray, r: np.ndarray) -> dict:
    """做多日減放空日的平均盤中報酬，OLS 對做多虛擬變數，Newey–West；單尾 > 0。"""
    keep = pos != 0
    pos, r = pos[keep], r[keep]
    n_long, n_short = int((pos > 0).sum()), int((pos < 0).sum())
    if min(n_long, n_short) < MIN_SIDE:
        return {"delta": np.nan, "t": np.nan, "p_one_sided": np.nan, "n": len(r), "n_long": n_long, "n_short": n_short}
    m = sm.OLS(r, sm.add_constant((pos > 0).astype(float))).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    t = float(m.tvalues[1])
    return {"delta": float(m.params[1]), "t": t, "p_one_sided": float(norm.sf(t)), "n": len(r),
            "n_long": n_long, "n_short": n_short}


def run_honhai(final: bool) -> None:
    cfg = load_config()
    out = output_path(cfg, "oos_2026", "honhai", "result.json")
    if final:
        guard_final(HONHAI_WINDOW, out)
    f = honhai_table(cfg)
    if not final:
        for tag, (first, last, folder) in USED.items():
            new = honhai_positions(f, first, last)
            ref = cfg["output_dir"] / HONHAI / "classifier_bert" / folder / "predictions.csv"
            if not ref.exists():
                print(f"{tag}：找不到 {ref}，跳過比對")
                continue
            old = pd.read_csv(ref, index_col=0, parse_dates=True)["A_logit"]
            both = new.index.intersection(old.index)
            print(f"{tag}：研究一 {len(old)} 天、面板版 {len(new)} 天、共同 {len(both)} 天；機率最大差 "
                  f"{(new.loc[both, 'prob'] - old[both]).abs().max():.1e}，多空方向相同 "
                  f"{(new.loc[both, 'pos'] == np.sign(old[both] - 0.5)).mean():.2%}")
        ev = f[(f["target_date"] >= HONHAI_WINDOW[0]) & (f["target_date"] <= HONHAI_WINDOW[1])]
        print(f"\n評估期 {HONHAI_WINDOW[0]} .. {HONHAI_WINDOW[1]}：目前有 {len(ev)} 天"
              f"（特徵齊全 {len(ev.dropna(subset=PRICE_FEATURES + ['ret_next']))} 天）")
        return

    d = honhai_positions(f, *HONHAI_WINDOW)
    pos, r = d["pos"].to_numpy(), d["r"].to_numpy()
    r1 = r1_test(pos, r)
    r1["pass"] = bool(r1["p_one_sided"] < 0.05)       # 做多或放空的日子不到 MIN_SIDE 天時是 NaN，算沒通過
    print(f"R1 2317 盤中：做多 {r1['n_long']} 天、放空 {r1['n_short']} 天，平均報酬差 {r1['delta']:+.4%}，"
          f"t = {r1['t']:+.2f}，單尾 p = {r1['p_one_sided']:.4f} -> {'通過' if r1['pass'] else '未通過'}\n")

    # 描述（不是假設）：隨機部位基準、成本後的回測
    log_r = np.log1p(d["r"])
    rand = random_sign_pctile(log_r, d["pos"])
    rows = {}
    for inst in ["stock", "futures"]:
        buy, sell, _ = cost_model(cfg, inst, daytrade=True)
        for name, position in [("多空", d["pos"]), ("每天放空", pd.Series(-1.0, index=d.index))]:
            bt = backtest(log_r, position, buy, sell, round_trip=True)
            rows[(inst, name)] = {**summary(bt), **trade_stats(bt, round_trip=True)}
    bts = pd.DataFrame(rows).T
    print(f"隨機部位基準（多空天數相同）：pctile {rand['pctile']:.3f}")
    print(bts[["total_return", "sharpe", "max_drawdown", "long_ratio", "breakeven_cost", "avg_cost"]].round(4))

    json.dump({"period": HONHAI_WINDOW, "R1": r1, "random_sign": rand,
               "backtest": {f"{i}_{n}": v for (i, n), v in rows.items()}},
              open(out, "w"), ensure_ascii=False, indent=2, default=float)
    d.to_csv(out.with_name("positions.csv"))
    print(f"\n-> {out}")


# ---------- 檢定力模擬 ----------
def _block_idx(rng, n_src: int, n: int, block: int = 20) -> np.ndarray:
    starts = rng.integers(0, n_src - block + 1, size=n // block + 1)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def _perm_p(rng, pos, r, reps: int = 500) -> float:
    """和 backtest.random_sign_pctile 相同的隨機部位檢定（向量化）。"""
    actual = (pos * r).mean()
    sims = (pos[np.argsort(rng.random((reps, len(pos))), axis=1)] * r).mean(axis=1)
    return float((sims >= actual).mean())


def run_power(sims: int = 600) -> None:
    """效果當真時的單尾 α = 0.05 檢定力。區塊 bootstrap（20 天一塊）從參考期間重抽評估期長度的樣本。
    「一半」：把效果縮成一半（V 把損失差減去一半的平均；R 把做多日的報酬往下移 δ/2），「零」：效果歸零，看型一錯誤。"""
    cfg = load_config()
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)

    rows = []
    for tag, k in [("開發期", "prereg_dev"), ("2024", "prereg_final"), ("2025–26", "replication")]:
        loss = pd.read_parquet(output_path(cfg, "volume", k, "cw_loss.parquet"))
        f = loss[[c for c in loss if c.startswith("y_lv_")]].dropna().mean(axis=1)
        ref = p.v.mean_test(f, NW_LAGS)
        for scale, s in [("全部", 1.0), ("一半", 0.5)]:
            g = (f - (1 - s) * f.mean()).to_numpy()
            for months, n in [(6, 124), (12, 245)]:
                et = s * ref["t"] * np.sqrt(n / ref["n"])
                ts = [p.v.mean_test(pd.Series(g[_block_idx(rng, len(g), n)]), NW_LAGS)["t"] for _ in range(sims)]
                rows.append({"效果來源": tag, "效果": scale, "月數": months, "天數": n, "解析式預期 t": et,
                             "解析式": norm.cdf(et - norm.ppf(0.95)), "bootstrap": (norm.sf(ts) < 0.05).mean()})
    print("V1 三檔平均 log 成交量")
    print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")

    f = honhai_table(cfg)
    refs = {tag: honhai_positions(f, first, last) for tag, (first, last, _) in USED.items()}
    refs["開發期＋2024"] = pd.concat(refs.values())
    print("R1 參考期間（研究一已用過）")
    for tag, d in refs.items():
        t = r1_test(d["pos"].to_numpy(), d["r"].to_numpy())
        print(f"  {tag}：{t['n']} 天，做多 {t['n_long']} 天，δ = {t['delta']:+.4%}，t = {t['t']:+.2f}")
    rows = []
    cases = [(t, "全部", 1.0) for t in refs] + [("開發期＋2024", "一半", 0.5), ("開發期＋2024", "零", 0.0)]
    for tag, scale, s in cases:
        d = refs[tag][refs[tag]["pos"] != 0]
        pos, r = d["pos"].to_numpy(), d["r"].to_numpy().copy()
        r[pos > 0] -= (1 - s) * (r[pos > 0].mean() - r[pos < 0].mean())
        for months, n in [(12, 245), (24, 490), (36, 735)]:
            res = []
            for _ in range(sims):
                i = _block_idx(rng, len(r), n)
                t = r1_test(pos[i], r[i])
                res.append((t["p_one_sided"] < 0.05, _perm_p(rng, pos[i], r[i]) < 0.05 if s == 0 else np.nan))
            res = np.array(res, dtype=float)
            rows.append({"效果來源": tag, "效果": scale, "月數": months, "天數": n, "NW t 檢定": res[:, 0].mean(),
                         "隨機部位檢定": np.nanmean(res[:, 1]) if s == 0 else np.nan})
    print("\nR1 2317 盤中（「零」那幾列是型一錯誤）")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["power", "volume", "honhai"])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.test == "power":
        run_power()
    elif a.test == "volume":
        run_volume(a.final)
    else:
        run_honhai(a.final)


if __name__ == "__main__":
    main()

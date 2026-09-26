"""[標的] 事件研究，第 3 步：開發期的探索性分析（不預先登記，結果要用 FDR 校正、標明是探索）。

    python scripts/09_target_explore.py

需要先跑 08_target_dev.py（output/target/dev/events_with_returns.parquet）。
「方向調整報酬」= 看多文的市場調整報酬，看空文取負號：> 0 表示作者說對。
每個特徵把事件分兩組，比較兩組的方向調整報酬（日期、股票雙向叢集），所有比較一起做 Benjamini-Hochberg。

會用到留言的特徵（推噓比、留言數、留言情緒）只用 entry 收盤（13:30）以前的留言，
而且只有 pttcc 有留言時間，所以只涵蓋 2019 年以後；文章最後的推噓總數含未來資訊，不用。
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import bert_path, load_config, output_path, work_path  # noqa: E402
from pttsent.prices import _roc_date  # noqa: E402

pd.set_option("display.width", 220)
TARGET_PRICE = re.compile(r"(?:目標價?|停利)[^\d\n]{0,6}(\d+(?:\.\d+)?)")
STOP_PRICE = re.compile(r"停損[^\d\n]{0,6}(\d+(?:\.\d+)?)")


def diff_test(d: pd.DataFrame, col: str, flag: pd.Series) -> dict:
    d = d.assign(_f=flag.astype(float)).dropna(subset=[col, "_f"])
    x = sm.add_constant(d["_f"].to_numpy())
    groups = np.column_stack([pd.factorize(d["e"])[0], pd.factorize(d["code"])[0]])
    fit = sm.OLS(d[col].to_numpy(), x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return {"n_yes": int(d["_f"].sum()), "n_no": int((1 - d["_f"]).sum()),
            "mean_yes": d.loc[d["_f"] == 1, col].mean(), "mean_no": d.loc[d["_f"] == 0, col].mean(),
            "diff": fit.params[1], "t": fit.tvalues[1], "p": float(fit.pvalues[1])}


def comments_before_entry(cfg, ev: pd.DataFrame) -> pd.DataFrame:
    """pttcc 留言中，時間早於 entry 收盤（13:30）的：每篇文章的推、噓、留言數與文字。"""
    ids = set(ev["article_id"])
    parts = []
    for y in range(2019, 2024):
        c = pd.read_parquet(work_path(cfg, "interim", "ptt", f"comments_{y}.parquet"),
                            columns=["article_id", "time", "tag", "content"])
        parts.append(c[c["article_id"].isin(ids)])
    c = pd.concat(parts).merge(ev[["article_id", "entry_date", "title"]], on="article_id")
    return c[c["time"] < c["entry_date"] + pd.Timedelta(hours=13, minutes=30)]


def news_days(cfg, codes) -> set:
    """鉅亨頭條被標記到的（代號, 日期）。"""
    out = set()
    for f in glob.glob(f"{cfg['data_dir']}/raw/cnyes/headline/20*/page_*.json"):
        if not ("2015" < f.split("/")[-2][:4] < "2024"):
            continue
        for a in json.load(open(f, encoding="utf-8"))["items"]["data"]:
            tags = set(a.get("stock") or []) & codes
            if tags:
                d = pd.Timestamp(a["publishAt"], unit="s", tz="Asia/Taipei").tz_localize(None).normalize()
                out |= {(c, d) for c in tags}
    return out


def punish_periods(cfg) -> pd.DataFrame:
    rows = []
    for p in glob.glob(f"{cfg['data_dir']}/raw/twse_punish/punish_*.json"):
        for r in json.load(open(p, encoding="utf-8")):
            a, _, b = str(r[6]).partition("～")
            try:
                rows.append((str(r[2]).strip(), _roc_date(a), _roc_date(b)))
            except (ValueError, TypeError):
                continue
    return pd.DataFrame(rows, columns=["code", "start", "end"])


def main():
    cfg = load_config()
    dev_dir = output_path(cfg, "target", "dev", "x").parent
    ev = pd.read_parquet(dev_dir / "events_with_returns.parquet")
    assert (ev["split"] == "dev").all()
    sign = np.where(ev["direction"] == "bullish", 1.0, -1.0)
    for h in (5, 20):
        ev[f"s_{h}"] = ev[f"car_{h}"] * sign
    posts = pd.read_parquet(work_path(cfg, "interim", "target_posts.parquet"), columns=["article_id", "content"])
    ev = ev.merge(posts, on="article_id", how="left")
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                            columns=["date", "code", "close", "high", "low", "value"])
    feats = {}

    # 1. 作者過去的準確度：只用 20 天窗口在這篇 entry 以前就結束的舊文
    ev = ev.sort_values("e").reset_index(drop=True)
    hit = pd.Series(np.nan, index=ev.index)
    n_prior = pd.Series(0, index=ev.index)
    for _, g in ev[ev["author"].notna()].groupby("author"):
        es_, ok = g["e"].to_numpy(), (g["s_20"] > 0).to_numpy(dtype=float)
        valid = g["s_20"].notna().to_numpy()
        for j, i in enumerate(g.index):
            prior = (es_ + 20 < es_[j]) & valid
            n_prior[i] = prior.sum()
            if prior.sum() >= 3:
                hit[i] = ok[prior].mean()
    feats["作者過去命中率 > 50%（至少 3 篇）"] = (hit > 0.5).where(hit.notna())

    # 2–3, 14. 進場前的留言
    c = comments_before_entry(cfg, ev)
    agg = c.groupby("article_id").agg(n=("tag", "size"), push=("tag", lambda s: (s == "push").sum()),
                                      boo=("tag", lambda s: (s == "boo").sum()))
    ev = ev.merge(agg, left_on="article_id", right_index=True, how="left")
    has_cc = ev["in_pttcc"] & (ev["entry_date"] >= "2019-01-01")
    ev.loc[has_cc, ["n", "push", "boo"]] = ev.loc[has_cc, ["n", "push", "boo"]].fillna(0)
    ratio = ev["push"] / (ev["push"] + ev["boo"])
    feats["進場前推噓比高於中位數"] = (ratio > ratio.median()).where(ratio.notna())
    feats["進場前留言數高於中位數"] = (ev["n"] > ev["n"].median()).where(ev["n"].notna())
    cache = work_path(cfg, "interim", "target_comment_scores.parquet")
    if not cache.exists():
        from pttsent.sentiment import bert
        model, tok = bert.load(bert_path(cfg))
        print(f"BERT 幫 {len(c):,} 則進場前留言打分……", flush=True)
        c = c.assign(score=bert.score(model, tok, c["content"], c["title"]))
        c[["article_id", "score"]].to_parquet(cache, index=False)
    sc = pd.read_parquet(cache).groupby("article_id")["score"].mean()
    agree = ev["article_id"].map(sc) * np.where(ev["direction"] == "bullish", 1, -1)
    feats["進場前留言情緒和作者同方向"] = (agree > 0).where(agree.notna() & (agree != 0))

    # 4–7. 文章本身
    body = ev["content"].fillna("")
    feats["內文長於中位數"] = body.str.len() > body.str.len().median()
    feats["有寫目標價或停利"] = body.str.contains(TARGET_PRICE)
    feats["附對帳單"] = ev["title"].str.contains("對帳單") | body.str.contains("對帳單")
    t = ev["time"]
    feats["盤中發文"] = (t.dt.normalize() == ev["entry_date"]) & (t.dt.hour >= 9)

    # 8–11. 股票與時期
    liq = panel.pivot_table(index="date", columns="code", values="value").rolling(20, min_periods=10).mean()
    lv = liq.stack().rename("liq")
    ev = ev.join(lv, on=["entry_date", "code"])
    feats["流動性最低三分之一"] = (ev["liq"] <= ev["liq"].quantile(1 / 3)).where(ev["liq"].notna())
    feats["上櫃股"] = ev["market"] == "TPEx"
    pun = punish_periods(cfg)
    in_pun = pd.Series(False, index=ev.index)
    for code, g in pun.groupby("code"):
        m = ev["code"] == code
        if m.any():
            d = ev.loc[m, "entry_date"]
            in_pun[m] = [((g["start"] <= x) & (g["end"] >= x)).any() for x in d]
    feats["進場時是處置股（僅上市）"] = in_pun.where(ev["market"] == "TWSE")
    feats["2021 年"] = ev["entry_date"].dt.year == 2021

    # 12. 被刪的文章（2019 以後才有意義：更早的 pttweb 幾乎全部標成已刪文）
    feats["被刪的文章（2019 以後）"] = ev["deleted"].where(ev["entry_date"] >= "2019-01-01")

    # 13. 發文前 3 天有這檔股票的鉅亨新聞
    nd = news_days(cfg, set(ev["code"]))
    day = t.dt.normalize()
    feats["發文前 3 天有鉅亨新聞"] = pd.Series([any((c_, d - pd.Timedelta(days=k)) in nd for k in range(4))
                                         for c_, d in zip(ev["code"], day)], index=ev.index)

    rows = []
    for name, f in feats.items():
        for h in (5, 20):
            rows.append({"特徵": name, "h": h, **diff_test(ev, f"s_{h}", f)})
    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    print("方向調整報酬：「是」組 − 「否」組（> 0 表示有這個特徵的文章比較準）")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    res.to_csv(dev_dir / "explore.csv", index=False)

    # 9. 大盤／台指文：之後加權報酬指數的方向調整報酬
    ix = pd.read_parquet(work_path(cfg, "processed", "target_index_events.parquet"))
    ix = ix[ix["split"] == "dev"]
    mk = pd.read_parquet(work_path(cfg, "prices", "market.parquet"))["TWSE"].to_numpy()
    for h in (5, 20):
        s = np.array([mk[e + 1:e + h + 1].sum() if e + h < len(mk) else np.nan for e in ix["e"]])
        s = s * np.where(ix["direction"] == "bullish", 1, -1)
        ok = ~np.isnan(s)
        fit = sm.OLS(s[ok], np.ones(ok.sum())).fit(cov_type="cluster", cov_kwds={"groups": ix["e"].to_numpy()[ok]})
        print(f"大盤文 {h} 天：{ok.sum()} 篇，方向調整報酬平均 {fit.params[0]:+.2%}，t = {fit.tvalues[0]:+.2f}，p = {fit.pvalues[0]:.3f}")

    # 8. 目標價與停損價：60 天內先碰到哪一個（只看數字合理的：目標在進場價 ±50% 內）
    px = panel.set_index(["code", "date"]).sort_index()
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    res_tp = []
    for _, r in ev.iterrows():
        tp, sl = TARGET_PRICE.search(r["content"] or ""), STOP_PRICE.search(r["content"] or "")
        if not (tp and sl) or (r["code"], r["entry_date"]) not in px.index:
            continue
        entry = px.loc[(r["code"], r["entry_date"]), "close"]
        tpv, slv = float(tp.group(1)), float(sl.group(1))
        bull = r["direction"] == "bullish"
        if not (0.5 * entry < tpv < 1.5 * entry and 0.5 * entry < slv < 1.5 * entry) \
                or (tpv > entry) != bull or (slv < entry) != bull:   # 目標和停損要在進場價的兩側
            continue
        i = days.get_loc(r["entry_date"])
        fwd = px.loc[r["code"]].reindex(days[i + 1:i + 61]).dropna()
        hit_tp = (fwd["high"] >= tpv) if bull else (fwd["low"] <= tpv)
        hit_sl = (fwd["low"] <= slv) if bull else (fwd["high"] >= slv)
        first_tp = hit_tp.idxmax() if hit_tp.any() else None
        first_sl = hit_sl.idxmax() if hit_sl.any() else None
        out = "都沒碰到" if not (first_tp or first_sl) else \
            "先到目標" if first_sl is None or (first_tp and first_tp < first_sl) else \
            "先到停損" if first_tp is None or first_sl < first_tp else "同一天"
        res_tp.append({"direction": r["direction"], "result": out, "tp_dist": abs(tpv / entry - 1), "sl_dist": abs(slv / entry - 1)})
    tpd = pd.DataFrame(res_tp)
    if len(tpd):
        print(f"\n目標價與停損價（{len(tpd)} 篇，目標距離中位數 {tpd.tp_dist.median():.1%}、停損距離中位數 {tpd.sl_dist.median():.1%}）")
        print(pd.crosstab(tpd["direction"], tpd["result"], normalize="index").round(3))
        tpd.to_csv(dev_dir / "target_stop.csv", index=False)
    print(f"\n結果 -> {dev_dir}")


if __name__ == "__main__":
    main()

"""大漲跌日前後的新聞與 PTT 討論（見 docs/large_moves.md）。只用開發期 2019–2023。

    python scripts/16_large_moves.py names     # 名稱比對第一輪：統計每個簡稱、列出要檢查的，抽樣寫成檢查表
    python scripts/16_large_moves.py recheck   # 第二輪：補了排除詞的簡稱重新抽樣
    python scripts/16_large_moves.py rules     # 兩輪都填完之後：每檔最後的比對規則
    python scripts/16_large_moves.py events    # 超額報酬、大漲跌日、後續事件、控制日、群聚日
    python scripts/16_large_moves.py windows   # 每個股票日五個窗口的新聞、PTT 則數與當沖比例（資料準備，不跑回歸）
    python scripts/16_large_moves.py analyze   # 第一部分與 P1–P4；程式和計畫都 commit 之後才會跑

輸出在 docs/large_moves/：
  names_stats.csv    每檔的統計與列入檢查的原因
  names_check.csv    第一輪檢查表；names_check2.csv 第二輪。ok 欄是 1（真的在講這家公司）或 0，
                     ok_claude 是 Claude 先標的一輪，note 是它不確定的理由；ok 由人工複核後定案
  names_precision.csv  每個簡稱、代號的正確率（第二輪有的用第二輪）
  names_rules.csv    每檔用不用簡稱、代號比對，以及排除詞
檢查表已經存在時不會覆寫（加 --force 才會）。檢查完的表和比對規則一起 commit，之後才算事件。

events 的輸出：data/interim/large_moves/days.parquet（每檔每個交易日一列，全期間）、
output/large_moves/events_summary.csv（事件期間內的個數）。
windows 的輸出：data/interim/large_moves/ptt_items.parquet（每檔被提到的文章與留言，第一次要跑十幾分鐘）、
windows.parquet（事件期間內每個股票日的窗口則數與控制變數）。
analyze 的輸出在 output/large_moves/：primary.csv（P1–P4 與 BH 校正）、part1.csv（第一部分），
以及拿掉群聚日的次要版本 *_nocluster.csv、各回歸完整係數 coef_*.csv。
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import large_moves as lm  # noqa: E402
from pttsent import volume as v  # noqa: E402
from pttsent.config import ROOT, load_config, output_path, work_path  # noqa: E402

DOC_DIR = ROOT / "docs" / "large_moves"
PTT_START = "2019-01-01"
NEWS_START = "2015-05-01"
FDR = 0.10
REGISTERED = ["docs/large_moves.md", "docs/large_moves", "scripts/16_large_moves.py", "src/pttsent/large_moves.py"]


def load_news_cached(cfg) -> pd.DataFrame:
    """鉅亨頭條讀一次要半分鐘，存一份 parquet。"""
    p = work_path(cfg, "interim", "large_moves", "news.parquet")
    if not p.exists():
        n = v.load_news(cfg["data_dir"])
        n.assign(stock=n["stock"].map(list)).to_parquet(p)
    n = pd.read_parquet(p)
    n["stock"] = n["stock"].map(tuple)
    return n


def dev_titles(cfg) -> pd.DataFrame:
    """開發期 PTT 文章標題，同一個標題只留第一則（回文的標題和原文一樣）。"""
    years = range(int(lm.DEV[0][:4]), int(lm.DEV[1][:4]))
    a = pd.concat([pq.read_table(work_path(cfg, "interim", "ptt", f"articles_{y}.parquet"), columns=["time", "title"]).to_pandas()
                   for y in years])
    a = a[(a["time"] >= lm.DEV[0]) & (a["time"] < lm.DEV[1])]
    a["key"] = a["title"].str.replace(r"^(Re|Fw):\s*", "", regex=True).str.strip()
    return a.sort_values("time").drop_duplicates("key")[["time", "title"]].reset_index(drop=True)


def write_new(df: pd.DataFrame, path: Path, force: bool):
    if path.exists() and not force:
        print(f"{path.name} 已經存在，不覆寫（加 --force）")
        return
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_doc(name: str) -> pd.DataFrame:
    return pd.read_csv(DOC_DIR / name, dtype={"code": str, "ok": str}, encoding="utf-8-sig", keep_default_na=False)


def panel_names(cfg) -> pd.DataFrame:
    return pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["name", "common"]).drop_duplicates()


def cmd_names(cfg, force):
    """第一輪：排除詞只用普通股名稱、沒有 EXTRA_EXCLUDE（照當時的程式，結果才重現得出來）。"""
    u = lm.universe(cfg["data_dir"])
    names = panel_names(cfg)
    excl = lm.superstrings(u, names.loc[names["common"], "name"].unique())
    news = load_news_cached(cfg)
    news = news[(news["time"] >= lm.DEV[0]) & (news["time"] < lm.DEV[1])]
    titles = dev_titles(cfg)

    stats = lm.flag_names(lm.name_stats(u, excl, news, titles["title"]))
    sheet = lm.check_sheet(stats, excl, titles)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(DOC_DIR / "names_stats.csv", index=False, encoding="utf-8-sig")
    write_new(sheet, DOC_DIR / "names_check.csv", force)

    flagged = stats[stats[["name_low_tag", "name_no_news", "code_year"]].any(axis=1)]
    print(f"PTT 標題 {len(titles):,} 則，鉅亨頭條 {len(news):,} 則")
    print(f"列入檢查：簡稱 {int((stats['name_low_tag'] | stats['name_no_news']).sum())} 個"
          f"（鉅亨標記比例低 {int(stats['name_low_tag'].sum())}、鉅亨太少 {int(stats['name_no_news'].sum())}），"
          f"像年份的代號 {int(stats['code_year'].sum())} 個；共 {len(flagged)} 檔")
    print(f"檢查表 {len(sheet)} 列 -> {DOC_DIR / 'names_check.csv'}")


def final_exclude(cfg) -> dict:
    u = lm.universe(cfg["data_dir"])
    return lm.superstrings(u, panel_names(cfg)["name"].unique(), lm.EXTRA_EXCLUDE)


def cmd_recheck(cfg, force):
    u = lm.universe(cfg["data_dir"])
    seen = read_doc("names_check.csv")[["code", "title"]]
    sheet = lm.recheck_sheet(u, final_exclude(cfg), dev_titles(cfg), seen, sorted(lm.EXTRA_EXCLUDE))
    write_new(sheet, DOC_DIR / "names_check2.csv", force)
    print(f"第二輪 {sheet['code'].nunique()} 個簡稱、{len(sheet)} 列 -> {DOC_DIR / 'names_check2.csv'}")


def cmd_rules(cfg, force):
    r1, r2 = read_doc("names_check.csv"), read_doc("names_check2.csv")
    if not (r1["ok"].isin(["0", "1"]).all() and r2.loc[r2["from_round1"] == 0, "ok"].isin(["0", "1"]).all()):
        sys.exit("還有 ok 沒填（只能是 0 或 1）")
    stats = read_doc("names_stats.csv")
    prec = lm.precision(r1, r2)
    prec.to_csv(DOC_DIR / "names_precision.csv", index=False, encoding="utf-8-sig")
    rules = lm.match_rules(stats, prec, final_exclude(cfg))
    rules.to_csv(DOC_DIR / "names_rules.csv", index=False, encoding="utf-8-sig")
    print(f"簡稱不用：{', '.join(rules.loc[~rules['ptt_name'], 'name'])}")
    print(f"代號不用：{', '.join(rules.loc[~rules['ptt_code'], 'code'])}")
    print(f"鉅亨只認股票標記：{int((~rules['news_title']).sum())} 檔")


def cmd_events(cfg, force):
    u = lm.universe(cfg["data_dir"])
    panel = pd.read_parquet(work_path(cfg, "prices", "panel.parquet"),
                            columns=["date", "code", "market", "close", "volume", "base", "ret", "mkt_ret",
                                     "limit_up_close", "limit_down_close"])
    days = pd.DatetimeIndex(sorted(panel.loc[panel["market"] == "TWSE", "date"].unique()))
    panel = panel[panel["code"].isin(set(u["code"]))]
    d = lm.stock_days(panel, days, v.load_disposition(cfg["data_dir"]), u["code"])
    d.to_parquet(work_path(cfg, "interim", "large_moves", "days.parquet"))

    p = d[d["in_period"]]
    rows = {
        "股票": p["code"].nunique(),
        "股票日": len(p),
        "排除：除權息或減資": int(p["adj"].sum()),
        "排除：停牌後恢復": int(p["halt"].sum()),
        "排除：處置期間": int(p["disp"].sum()),
        "排除：β 或 σ 不足、報酬缺值": int((p["ar"].isna() | p["sigma"].isna()).sum()),
        "合格": int(p["eligible"].sum()),
        "事件（全部）": int(p["event"].sum()),
        "後續事件": int(p["followon"].sum()),
        "主要事件": int(p["main"].sum()),
        "主要：大漲": int((p["main"] & p["up"]).sum()),
        "主要：大跌": int((p["main"] & ~p["up"]).sum()),
        "主要：3–5σ": int((p["main"] & (p["z"].abs() < 5)).sum()),
        "主要：5σ 以上": int((p["main"] & (p["z"].abs() >= 5)).sum()),
        "主要：群聚日": int((p["main"] & p["cluster"]).sum()),
        "主要：收盤鎖漲跌停": int((p["main"] & (p["limit_up_close"] | p["limit_down_close"])).sum()),
        "控制日": int(p["control"].sum()),
    }
    s = pd.Series(rows, name="n")
    s.to_csv(output_path(cfg, "large_moves", "events_summary.csv"), encoding="utf-8-sig")
    print(s.to_string())
    m = p[p["main"]]
    print("\n主要事件，依年份與方向：")
    print(pd.crosstab(m["date"].dt.year, m["up"].map({True: "大漲", False: "大跌"})).to_string())
    print("\n每檔主要事件數：", m.groupby("code").size().reindex(u["code"], fill_value=0).describe().round(1).to_dict())


def load_rules() -> pd.DataFrame:
    r = read_doc("names_rules.csv")
    for c in ["ptt_name", "ptt_code", "news_title", "name_low_tag"]:
        r[c] = r[c].map({"True": True, "False": False}) if r[c].dtype == object else r[c].astype(bool)
    return r


def load_daytrade(data_dir, end) -> pd.Series:
    """證交所 TWTB4U：(日期, 代號) -> 當沖成交股數。只有上市股票。"""
    rows = []
    for f in sorted(glob.glob(f"{data_dir}/raw/twse_twtb4u/twtb4u_*.json")):
        day = pd.Timestamp(Path(f).stem.split("_")[1])
        if not (pd.Timestamp(PTT_START) - pd.Timedelta(days=60) <= day < pd.Timestamp(end)):
            continue
        for r in json.load(open(f, encoding="utf-8")):
            rows.append((day, str(r[0]).strip(), float(str(r[3]).replace(",", "") or "nan")))
    s = pd.DataFrame(rows, columns=["date", "code", "dt"]).drop_duplicates(["date", "code"])
    return s.set_index(["date", "code"])["dt"]


def cmd_windows(cfg, force):
    rules = load_rules()
    days = pd.DatetimeIndex(sorted(pd.read_parquet(work_path(cfg, "prices", "panel.parquet"), columns=["date", "market"])
                                   .query("market == 'TWSE'")["date"].unique()))
    d = pd.read_parquet(work_path(cfg, "interim", "large_moves", "days.parquet"))
    end = lm.DEV[1]

    items_path = work_path(cfg, "interim", "large_moves", "ptt_items.parquet")
    if force or not items_path.exists():
        years = range(int(PTT_START[:4]), int(end[:4]))
        ptt = lm.ptt_items(work_path(cfg, "interim", "ptt", "x").parent, years, rules,
                           cfg["sentiment"]["max_comment_lag_days"], end)
        ptt.to_parquet(items_path)
    ptt = pd.read_parquet(items_path)
    news = load_news_cached(cfg)
    news = news[(news["time"] >= NEWS_START) & (news["time"] < end)]
    news = lm.news_items(news, rules)
    print(f"PTT 文章＋留言 {len(ptt):,} 則，鉅亨新聞（每檔分開算）{len(news):,} 則")

    # 前一天、下一天要用同一檔自己的列，所以先在整個期間算，再切事件期間
    d = d[d["date"] < end].copy()
    dt = load_daytrade(cfg["data_dir"], end)
    key = pd.MultiIndex.from_arrays([d["date"], d["code"]])
    d["dt_intraday"] = (dt.reindex(key).to_numpy() / d["volume"].to_numpy())
    d.loc[d["market"] != "TWSE", "dt_intraday"] = np.nan
    d["locked"] = d["limit_up_close"] | d["limit_down_close"]
    d = lm.stock_lags(d)
    for k in (5, 20):
        d[f"scar{k}"] = lm.scar(d, k, end)

    d = d[d["in_period"]].copy()
    d = lm.add_windows(d, news, days, "news", start=NEWS_START, end=end)
    d = lm.add_windows(d, ptt, days, "ptt", start=PTT_START, end=end)
    d = lm.outcomes(d)
    d["code_year"] = d["code"] + "_" + d["date"].dt.year.astype(str)
    d["year_month"] = d["date"].dt.to_period("M").astype(str)
    d.to_parquet(work_path(cfg, "interim", "large_moves", "windows.parquet"))
    print(f"windows.parquet：{len(d):,} 列；主要事件 {int(d['main'].sum())}、控制日 {int(d['control'].sum())}")
    print("缺值比例：", d[["news_preopen", "ptt_preopen", "ptt_base", "dt_intraday", "scar5", "scar20"]]
          .isna().mean().round(4).to_dict())


def committed(paths) -> bool:
    r = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def coef_table(f, name) -> pd.DataFrame:
    ci = f.conf_int()
    return pd.DataFrame({"model": name, "coef": f.params, "se": f.bse, "t": f.tvalues, "p": f.pvalues,
                         "lo": ci[0], "hi": ci[1], "n": int(f.nobs)})


def primary(d: pd.DataFrame, tag: str, cfg) -> pd.DataFrame:
    f12 = lm.p12(d)
    f5, f20 = lm.p34(d, 5), lm.p34(d, 20)
    for f, name in ((f12, "p12"), (f5, "p3"), (f20, "p4")):
        coef_table(f, name).to_csv(output_path(cfg, "large_moves", f"coef_{name}{tag}.csv"), encoding="utf-8-sig")
    rows = [
        ("P1", "開盤前 PTT 異常 → 事件", f12, "ptt_abn_preopen", "greater"),
        ("P2", "事前 PTT 異常 → 事件", f12, "ptt_abn_pre5", "greater"),
        ("P3", "無新聞 → sCAR(5)", f5, "nonews", "two"),
        ("P4", "無新聞 → sCAR(20)", f20, "nonews", "two"),
    ]
    out = []
    for k, desc, f, x, side in rows:
        tv = f.tvalues[x]
        p = 1 - norm.cdf(tv) if side == "greater" else 2 * (1 - norm.cdf(abs(tv)))
        out.append({"test": k, "desc": desc, "coef": f.params[x], "se": f.bse[x], "t": tv, "p": p,
                    "side": side, "n": int(f.nobs)})
    out = pd.DataFrame(out)
    rej, p_fdr, _, _ = multipletests(out["p"], alpha=FDR, method="fdr_bh")
    out["p_fdr"], out["pass"] = p_fdr, rej
    return out


def cmd_analyze(cfg, force):
    if not committed(REGISTERED):
        sys.exit("計畫或程式有沒 commit 的修改；照計畫要先 commit 再跑")
    d = pd.read_parquet(work_path(cfg, "interim", "large_moves", "windows.parquet"))
    ys = [f"{k}_{w}" for w in lm.WINDOWS for k in ("news_any", "news_log", "ptt_abn")] + ["dt_intraday", "dt_next"]
    nocl = d[~(d["main"] & d["cluster"])]
    for data, tag in ((d, ""), (nocl, "_nocluster")):
        lm.part1(data, ys).to_csv(output_path(cfg, "large_moves", f"part1{tag}.csv"), index=False, encoding="utf-8-sig")
        pr = primary(data, tag, cfg)
        pr.to_csv(output_path(cfg, "large_moves", f"primary{tag}.csv"), index=False, encoding="utf-8-sig")
        print(f"\n主要檢定{'（拿掉群聚日，次要）' if tag else ''}：")
        print(pr.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["names", "recheck", "rules", "events", "windows", "analyze"])
    ap.add_argument("--force", action="store_true",
                    help="names/recheck：覆寫檢查表；windows：重算 PTT 比對")
    args = ap.parse_args()
    cfg = load_config()
    {"names": cmd_names, "recheck": cmd_recheck, "rules": cmd_rules, "events": cmd_events,
     "windows": cmd_windows, "analyze": cmd_analyze}[args.cmd](cfg, args.force)


if __name__ == "__main__":
    main()

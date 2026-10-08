"""大漲跌日前後的新聞與 PTT 討論（見 docs/large_moves.md）。只用開發期 2019–2023。

    python scripts/16_large_moves.py names     # 名稱比對第一輪：統計每個簡稱、列出要檢查的，抽樣寫成檢查表
    python scripts/16_large_moves.py recheck   # 第二輪：補了排除詞的簡稱重新抽樣
    python scripts/16_large_moves.py rules     # 兩輪都填完之後：每檔最後的比對規則
    python scripts/16_large_moves.py events    # 超額報酬、大漲跌日、後續事件、控制日、群聚日

輸出在 docs/large_moves/：
  names_stats.csv    每檔的統計與列入檢查的原因
  names_check.csv    第一輪檢查表；names_check2.csv 第二輪。ok 欄是 1（真的在講這家公司）或 0，
                     ok_claude 是 Claude 先標的一輪，note 是它不確定的理由；ok 由人工複核後定案
  names_precision.csv  每個簡稱、代號的正確率（第二輪有的用第二輪）
  names_rules.csv    每檔用不用簡稱、代號比對，以及排除詞
檢查表已經存在時不會覆寫（加 --force 才會）。檢查完的表和比對規則一起 commit，之後才算事件。

events 的輸出：data/interim/large_moves/days.parquet（每檔每個交易日一列，全期間）、
output/large_moves/events_summary.csv（事件期間內的個數）。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import large_moves as lm  # noqa: E402
from pttsent import volume as v  # noqa: E402
from pttsent.config import ROOT, load_config, output_path, work_path  # noqa: E402

DOC_DIR = ROOT / "docs" / "large_moves"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["names", "recheck", "rules", "events"])
    ap.add_argument("--force", action="store_true", help="覆寫已經存在的檢查表")
    args = ap.parse_args()
    cfg = load_config()
    {"names": cmd_names, "recheck": cmd_recheck, "rules": cmd_rules, "events": cmd_events}[args.cmd](cfg, args.force)


if __name__ == "__main__":
    main()

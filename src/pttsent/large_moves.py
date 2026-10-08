"""大漲跌日前後的新聞與 PTT 討論（見 docs/large_moves.md）。

這裡先放名稱比對：267 檔的簡稱怎麼比對到 PTT 標題與鉅亨新聞，以及哪些簡稱要人工檢查。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEV = ("2019-01-01", "2024-01-01")   # 名稱檢查只用開發期的文字
YEAR_CODES = (1990, 2030)            # 這個範圍的代號會撞到西元年份（「2008年」）
TAG_FRAC_MIN, TAG_MIN_HITS = 0.8, 5  # 鉅亨標題提到簡稱、而且標記了這檔的比例低於 0.8，就列入人工檢查
PTT_MIN_HITS = 10                    # 鉅亨標題太少、無法估計的兩字簡稱，PTT 標題至少這麼多則才列入
PRECISION_MIN = 0.8                  # 人工檢查的正確率低於這個，就不用這個簡稱（或代號）比對

# 第一輪檢查後補的排除詞：錯誤集中在固定幾種寫法的簡稱。補了之後另外抽一批沒看過的標題重新檢查（第二輪）
EXTRA_EXCLUDE = {
    "1210": ["大成長", "大成立", "大成交", "大成功"],
    "1216": ["統一投信", "統一投顧", "統一證", "統一期貨", "統一FANG", "國家統一", "兩德統一", "規格統一", "統一用"],
    "1303": ["東南亞"],
    "1414": ["東和鋼", "美東和"],
    "1423": ["華爾街"],
    "1503": ["滬士電"],
    "1515": ["壓力山大"],
    "1519": ["京華城", "三星華城"],
    "1603": ["華電子", "華電動"],
    "1612": ["宏泰人壽"],
    "1713": ["美國化", "中國化"],
    "1810": ["溫和成長", "和成交"],
    "2106": ["建大型"],
    "2312": ["金寶座"],                      # 泰金寶-DR 由 superstrings 處理
    "2329": ["美華泰"],
    "2605": ["新興市場", "新興科技", "新興技術", "新興產業", "新興亞", "新興國家", "新興記憶體"],
    "2905": ["三商美邦", "三商人壽"],
}


def norm_text(s: pd.Series) -> pd.Series:
    """臺 -> 台；比對前文字和簡稱都先做。"""
    return s.str.replace("臺", "台", regex=False)


def norm_name(name: str) -> str:
    """簡稱 -> 一般寫法：去掉「*」「-KY」「-DR」，臺 -> 台。"""
    name = re.sub(r"\*$", "", name.strip())
    name = re.sub(r"-(KY|DR)$", "", name, flags=re.I)
    return name.replace("臺", "台")


def universe(data_dir) -> pd.DataFrame:
    u = pd.read_csv(Path(data_dir) / "universe_267.csv", dtype=str, encoding="utf-8-sig")
    u["name"] = u["name_short"].map(norm_name)
    return u.rename(columns={"ticker": "code"})[["code", "name", "name_short", "sector"]]


def superstrings(u: pd.DataFrame, all_names, extra: dict | None = None) -> dict:
    """每檔的排除詞：包含它簡稱的其他名稱（「南亞」->「南亞科」），加上 extra；比對前先從文字刪掉。

    第一輪檢查只用了普通股的名稱，漏了存託憑證與 ETF（「泰金寶-DR」「統一FANG+」）；之後一律用面板裡所有名稱。
    """
    others = {norm_name(n) for n in all_names}
    extra = extra or {}
    return {c: sorted({x for x in others if n in x and x != n} | set(extra.get(c, [])))
            for c, n in zip(u["code"], u["name"])}


def remove_words(s: pd.Series, words) -> pd.Series:
    for w in words:
        s = s.str.replace(w, "", regex=False)
    return s


def code_regex(code: str) -> str:
    return f"(?:^|[^0-9]){code}(?:[^0-9]|$)"


def name_stats(u: pd.DataFrame, excl: dict, news: pd.DataFrame, titles: pd.Series) -> pd.DataFrame:
    """每個簡稱：鉅亨標題提到幾則、其中幾則標記了這檔（tag_frac），PTT 標題提到幾則。"""
    nt = norm_text(news["title"])
    pt = norm_text(titles)
    rows = []
    for c, n in zip(u["code"], u["name"]):
        hit = remove_words(nt, excl[c]).str.contains(n, regex=False)
        tagged = news["stock"].map(lambda s: c in s)
        rows.append({"code": c, "name": n, "exclude": "/".join(excl[c]),
                     "news_title": int(hit.sum()), "news_title_tagged": int((hit & tagged).sum()),
                     "news_tag": int(tagged.sum()),
                     "ptt_title_name": int(remove_words(pt, excl[c]).str.contains(n, regex=False).sum()),
                     "ptt_title_code": int(pt.str.contains(code_regex(c)).sum())})
    d = pd.DataFrame(rows)
    d["tag_frac"] = d["news_title_tagged"] / d["news_title"].where(d["news_title"] > 0)
    return d


def flag_names(d: pd.DataFrame) -> pd.DataFrame:
    """要人工檢查的簡稱與代號，以及原因。

    name_low_tag  鉅亨標題提到簡稱、但多半沒標記這檔：簡稱常被當成別的意思用
    name_no_news  鉅亨標題太少、估不出來的兩字簡稱，但 PTT 標題常出現
    code_year     代號像西元年份
    """
    low = (d["news_title"] >= TAG_MIN_HITS) & (d["tag_frac"] < TAG_FRAC_MIN)
    no_news = (d["news_title"] < TAG_MIN_HITS) & (d["ptt_title_name"] >= PTT_MIN_HITS) & (d["name"].str.len() <= 2)
    year = d["code"].astype(int).between(*YEAR_CODES)
    out = d.assign(name_low_tag=low, name_no_news=no_news, code_year=year)
    return out


def check_sheet(d: pd.DataFrame, excl: dict, titles: pd.DataFrame, n: int = 30, seed: int = 42) -> pd.DataFrame:
    """人工檢查用的抽樣：每個列出的簡稱抽 n 則提到它（但沒有代號）的 PTT 標題，每個像年份的代號抽 n 則提到代號的。

    titles：PTT 文章的 time、title（開發期、已去重）。`ok` 欄留白，人工填 1（真的在講這家公司）或 0。
    """
    pt = norm_text(titles["title"])
    rng = np.random.default_rng(seed)
    out = []
    for r in d.itertuples():
        jobs = []
        if r.name_low_tag or r.name_no_news:
            hit = remove_words(pt, excl[r.code]).str.contains(r.name, regex=False) & ~pt.str.contains(code_regex(r.code))
            jobs.append(("name", hit))
        if r.code_year:
            jobs.append(("code", pt.str.contains(code_regex(r.code))))
        for kind, hit in jobs:
            idx = np.flatnonzero(hit.to_numpy())
            pick = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else idx
            s = titles.iloc[pick]
            out.append(pd.DataFrame({"code": r.code, "name": r.name, "kind": kind, "n_hits": len(idx),
                                     "time": s["time"].dt.strftime("%Y-%m-%d"), "title": s["title"], "ok": ""}))
    return pd.concat(out, ignore_index=True)


def recheck_sheet(u: pd.DataFrame, excl: dict, titles: pd.DataFrame, seen: pd.DataFrame, codes, n: int = 30,
                  seed: int = 42) -> pd.DataFrame:
    """第二輪：補了排除詞的簡稱，重新檢查。

    補了之後還比對得到的標題不超過 n 則，就全部檢查（等於整個開發期的正確率），其中第一輪看過的沿用第一輪的標註
    （from_round1 = 1，ok 留白，算正確率時去第一輪找）；超過 n 則，從第一輪沒看過的標題抽 n 則。每檔用自己的亂數種子。
    seen：第一輪檢查表的 code、title；「看過」是指第一輪在同一檔底下檢查過。
    """
    pt = norm_text(titles["title"])
    names = u.set_index("code")["name"]
    out = []
    for c in codes:
        was_seen = titles["title"].isin(set(seen.loc[seen["code"] == c, "title"])).to_numpy()
        hit = (remove_words(pt, excl[c]).str.contains(names[c], regex=False) & ~pt.str.contains(code_regex(c))).to_numpy()
        idx = np.flatnonzero(hit)
        if len(idx) > n:
            rng = np.random.default_rng([seed, int(c)])
            idx = np.sort(rng.choice(np.flatnonzero(hit & ~was_seen), size=n, replace=False))
        s = titles.iloc[idx]
        out.append(pd.DataFrame({"code": c, "name": names[c], "kind": "name", "n_hits": int(hit.sum()),
                                 "time": s["time"].dt.strftime("%Y-%m-%d"), "title": s["title"],
                                 "from_round1": was_seen[idx].astype(int), "ok": ""}))
    return pd.concat(out, ignore_index=True)


def precision(r1: pd.DataFrame, r2: pd.DataFrame) -> pd.DataFrame:
    """每個（代號, 簡稱或代號）的人工檢查正確率；第二輪有的用第二輪（補了排除詞之後）。

    第二輪 from_round1 = 1 的列去第一輪找同一個標題的 ok。第二輪一列都沒有的簡稱（補了排除詞就比對不到了）
    正確率記為缺值，不影響規則。
    """
    first = r1.drop_duplicates(["code", "title"]).set_index(["code", "title"])["ok"]
    r2 = r2.copy()
    carried = r2["from_round1"].astype(int) == 1
    r2.loc[carried, "ok"] = [first[(c, t)] for c, t in zip(r2.loc[carried, "code"], r2.loc[carried, "title"])]
    rows = []
    for rnd, s in enumerate([r1, r2], 1):
        g = s.assign(ok=pd.to_numeric(s["ok"])).groupby(["code", "kind"])["ok"].agg(["mean", "size"])
        rows.append(g.assign(round=rnd))
    d = pd.concat(rows).reset_index().sort_values("round").drop_duplicates(["code", "kind"], keep="last")
    redone = {(c, "name") for c in EXTRA_EXCLUDE} - set(zip(r2["code"], r2["kind"]))
    d.loc[[(c, k) in redone for c, k in zip(d["code"], d["kind"])], ["mean", "size", "round"]] = [np.nan, 0, 2]
    return d.rename(columns={"mean": "precision", "size": "n_checked"}).reset_index(drop=True)


def match_rules(stats: pd.DataFrame, prec: pd.DataFrame, excl: dict) -> pd.DataFrame:
    """每檔最後的比對規則。

    ptt_name     PTT 用簡稱比對：沒列入檢查，或檢查的正確率 ≥ PRECISION_MIN
    ptt_code     PTT 用代號比對：同上（只有像年份的代號檢查過）
    news_title   鉅亨用標題裡的簡稱比對：鉅亨標記比例沒有偏低，而且 PTT 也可以用簡稱；否則只認股票標記
    """
    p = prec.set_index(["code", "kind"])["precision"]
    out = stats[["code", "name", "name_low_tag", "name_no_news", "code_year"]].copy()
    out["name_precision"] = [p.get((c, "name"), np.nan) for c in out["code"]]
    out["code_precision"] = [p.get((c, "code"), np.nan) for c in out["code"]]
    out["ptt_name"] = out["name_precision"].isna() | (out["name_precision"] >= PRECISION_MIN)
    out["ptt_code"] = out["code_precision"].isna() | (out["code_precision"] >= PRECISION_MIN)
    out["news_title"] = ~out["name_low_tag"] & out["ptt_name"]
    out["exclude"] = ["/".join(excl[c]) for c in out["code"]]
    return out

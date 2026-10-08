"""大漲跌日前後的新聞與 PTT 討論（見 docs/large_moves.md）。

名稱比對：267 檔的簡稱怎麼比對到 PTT 標題與鉅亨新聞，以及哪些簡稱要人工檢查。
事件：超額報酬、大漲跌日、後續事件、控制日、群聚日。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .volume import in_periods

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


# ---- 事件 ----

EVENT_START, EVENT_END = "2019-03-01", "2023-11-30"   # 事件日 t 的範圍
BETA_WIN, BETA_MIN = 250, 120                         # β：前 250 個交易日（不含 t），至少 120 天
SIG_WIN, SIG_MIN = 60, 40                             # σ：前 60 個交易日的 ar 標準差，至少 40 天
Z_MIN, AR_MIN = 3.0, 0.03                             # 大漲跌日：|ar| ≥ 3σ 而且 |ar| ≥ 3%
FOLLOW = 5          # 前一個事件之後這麼多個交易日內的事件算後續事件
CONTROL_GAP = 5     # 控制日前後這麼多個交易日內不能有任何事件
HALT_GAP = 5        # 和這檔上一個交易日相隔超過這麼多個市場交易日，算停牌後恢復交易
CLUSTER_FRAC = 0.05  # 同一天同方向事件佔當天合格股票的比例達到這個，算群聚日


def market_model(ret: pd.Series, mkt: pd.Series) -> pd.DataFrame:
    """一檔股票（依日期排序）的 β、超額報酬 ar 與 σ。β 與 σ 都只用 t 之前的列。"""
    ok = ret.notna() & mkt.notna()
    x, y = mkt.where(ok), ret.where(ok)

    def past_mean(v):
        return v.rolling(BETA_WIN, min_periods=BETA_MIN).mean().shift(1)

    mx, my = past_mean(x), past_mean(y)
    beta = (past_mean(x * y) - mx * my) / (past_mean(x * x) - mx ** 2)
    ar = ret - beta * mkt
    sigma = ar.rolling(SIG_WIN, min_periods=SIG_MIN).std().shift(1)
    return pd.DataFrame({"beta": beta, "ar": ar, "sigma": sigma})


def follow_on(pos: np.ndarray, gap: int = FOLLOW) -> np.ndarray:
    """事件的交易日位置（遞增）-> 是否在前一個事件之後 gap 個交易日內。"""
    d = np.diff(pos, prepend=-10 ** 9)
    return d <= gap


def near_event(pos: np.ndarray, event_pos: np.ndarray, gap: int = CONTROL_GAP) -> np.ndarray:
    """每個位置前後 gap 個交易日內（含當天）有沒有事件。event_pos 要遞增。"""
    if len(event_pos) == 0:
        return np.zeros(len(pos), dtype=bool)
    lo = np.searchsorted(event_pos, pos - gap, side="left")
    hi = np.searchsorted(event_pos, pos + gap, side="right")
    return hi > lo


def stock_days(panel: pd.DataFrame, days: pd.DatetimeIndex, disp: pd.DataFrame, codes) -> pd.DataFrame:
    """所有股票、所有交易日（全期間，切期間在最後）的超額報酬與事件旗標。

    panel：面板裡這些股票的列；days：市場交易日曆（上市）；disp：處置期間（code、start、end）。
    除權息、減資恢復買賣：基準價和前一天收盤不同的日子（兩個市場通用）。
    """
    pos_of = pd.Series(np.arange(len(days)), index=days)
    out = []
    for c in codes:
        px = panel[panel["code"] == c].sort_values("date").reset_index(drop=True)
        if px.empty:
            continue
        px = px[px["date"].isin(days)].reset_index(drop=True)
        px = pd.concat([px, market_model(px["ret"], px["mkt_ret"])], axis=1)
        px["pos"] = pos_of.reindex(px["date"]).to_numpy()
        prev_close = px["close"].shift(1)
        px["adj"] = (px["base"] / prev_close - 1).abs() > 1e-6
        px["halt"] = px["pos"].diff() > HALT_GAP
        px["disp"] = in_periods(pd.DatetimeIndex(px["date"]), disp[disp["code"] == c]).to_numpy() > 0
        px["eligible"] = px["ar"].notna() & px["sigma"].notna() & ~px["adj"] & ~px["halt"] & ~px["disp"]
        px["z"] = px["ar"] / px["sigma"]
        px["event"] = px["eligible"] & (px["z"].abs() >= Z_MIN) & (px["ar"].abs() >= AR_MIN)
        ev = px.index[px["event"]]
        px["followon"] = False
        px.loc[ev, "followon"] = follow_on(px.loc[ev, "pos"].to_numpy())
        px["control"] = px["eligible"] & ~near_event(px["pos"].to_numpy(), px.loc[ev, "pos"].to_numpy())
        out.append(px)
    d = pd.concat(out, ignore_index=True)
    d["up"] = d["ar"] > 0
    d["main"] = d["event"] & ~d["followon"]

    # 群聚日：同一天、同方向的事件（含後續事件）佔當天合格股票的比例
    g = d[d["eligible"]].groupby("date")
    n_ok = g.size()
    frac_up = d[d["event"] & d["up"]].groupby("date").size().reindex(n_ok.index, fill_value=0) / n_ok
    frac_dn = d[d["event"] & ~d["up"]].groupby("date").size().reindex(n_ok.index, fill_value=0) / n_ok
    cu = d["date"].map(frac_up >= CLUSTER_FRAC).fillna(False).astype(bool)
    cd = d["date"].map(frac_dn >= CLUSTER_FRAC).fillna(False).astype(bool)
    d["cluster"] = d["event"] & ((d["up"] & cu) | (~d["up"] & cd))
    d["in_period"] = (d["date"] >= EVENT_START) & (d["date"] <= EVENT_END)
    return d

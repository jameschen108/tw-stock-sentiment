"""大漲跌日前後的新聞與 PTT 討論（見 docs/large_moves.md）。

名稱比對：267 檔的簡稱怎麼比對到 PTT 標題與鉅亨新聞，以及哪些簡稱要人工檢查。
事件：超額報酬、大漲跌日、後續事件、控制日、群聚日。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import assign_trade_date
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


# ---- 時間窗 ----

OPEN = 9                     # 開盤 09:00；收盤 13:30 由 assign_trade_date 處理
WINDOWS = {"pre5": 5, "preopen": 1, "intraday": 1, "next": 1, "post": 4}   # 窗口 -> 天數
BASE = (25, 6)               # PTT 基準期：t−25 到 t−6 這 20 個交易日


def day_parts(times: pd.Series, days: pd.DatetimeIndex) -> pd.DataFrame:
    """每則貼文或新聞 -> 所屬交易日的位置（前一天收盤後到當天收盤）、是不是在開盤前。

    索引是輸入的位置（0, 1, ...）；超出日曆的列丟掉。
    """
    t = pd.to_datetime(times.reset_index(drop=True))
    td = assign_trade_date(t, days)
    ok = td.notna()
    t, td = t[ok], td[ok]
    return pd.DataFrame({"pos": days.searchsorted(td), "pre": (t < td + pd.Timedelta(hours=OPEN)).to_numpy()},
                        index=t.index)


def daily_counts(parts: pd.DataFrame, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """每個交易日開盤前、盤中（含前一天收盤後到午夜這段算開盤前）的則數。"""
    pre = np.bincount(parts.loc[parts["pre"], "pos"], minlength=n_days)[:n_days].astype(float)
    intra = np.bincount(parts.loc[~parts["pre"], "pos"], minlength=n_days)[:n_days].astype(float)
    return pre, intra


def window_counts(pre: np.ndarray, intra: np.ndarray, i: np.ndarray) -> pd.DataFrame:
    """事件日位置 i 的五個窗口的則數，以及基準期每日 log1p 則數的平均。超出陣列範圍的是缺值。"""
    full = pre + intra
    n = len(full)
    cs = np.concatenate([[0.0], np.cumsum(full)])
    lcs = np.concatenate([[0.0], np.cumsum(np.log1p(full))])

    def rng_sum(c, a, b):        # 位置 a..b（含）的總和
        out = np.full(len(i), np.nan)
        ok = (a >= 0) & (b < n)
        out[ok] = c[b[ok] + 1] - c[a[ok]]
        return out

    def at(v, k):
        out = np.full(len(i), np.nan)
        ok = (k >= 0) & (k < n)
        out[ok] = v[k[ok]]
        return out

    return pd.DataFrame({
        "pre5": rng_sum(cs, i - 5, i - 1),
        "preopen": at(pre, i),
        "intraday": at(intra, i),
        "next": at(full, i + 1),
        "post": rng_sum(cs, i + 2, i + 5),
        "base": rng_sum(lcs, i - BASE[0], i - BASE[1]) / (BASE[0] - BASE[1] + 1),
    })


# ---- 比對：PTT 與鉅亨 ----

def _has_latin(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s))


def mentions(text: pd.Series, rule) -> np.ndarray:
    """text（已經 norm_text）裡提到這檔的列。rule：names_rules 的一列（code、name、ptt_name、ptt_code、exclude）。"""
    hit = np.zeros(len(text), dtype=bool)
    if rule.ptt_code:
        cand = text.str.contains(rule.code, regex=False).to_numpy()
        if cand.any():
            hit[cand] = text[cand].str.contains(code_regex(rule.code)).to_numpy()
    if rule.ptt_name:
        case = not _has_latin(rule.name)
        cand = text.str.contains(rule.name, case=case, regex=False).to_numpy() & ~hit
        if cand.any():
            excl = [w for w in str(rule.exclude or "").split("/") if w]
            sub = remove_words(text[cand], excl) if excl else text[cand]
            hit[cand] = sub.str.contains(rule.name, case=case, regex=False).to_numpy()
    return hit


def ptt_items(ptt_dir: Path, years, rules: pd.DataFrame, max_lag_days: int, end: str) -> pd.DataFrame:
    """每檔被提到的 PTT 文章與留言（code、time）。比照研究一的 select_texts：
    文章算標題提到的；留言算在這些文章底下的，或留言本身提到的。留言晚於文章超過 max_lag_days 的不算。
    """
    import pyarrow.parquet as pq
    out = []
    for y in years:
        a = pq.read_table(ptt_dir / f"articles_{y}.parquet", columns=["article_id", "time", "title"]).to_pandas()
        c = pq.read_table(ptt_dir / f"comments_{y}.parquet", columns=["article_id", "time", "content", "lag_sec"]).to_pandas()
        a, c = a[a["time"] < end], c[(c["time"] < end) & (c["lag_sec"] <= max_lag_days * 86400)]
        title = norm_text(a["title"].fillna(""))
        content = norm_text(c["content"].fillna(""))
        for r in rules.itertuples():
            ta = mentions(title, r)
            ids = set(a.loc[ta, "article_id"])
            tc = c["article_id"].isin(ids).to_numpy() | mentions(content, r)
            out.append(pd.DataFrame({"code": r.code, "time": np.concatenate([a.loc[ta, "time"].to_numpy(),
                                                                              c.loc[tc, "time"].to_numpy()])}))
    return pd.concat(out, ignore_index=True)


def news_items(news: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """每檔被提到的鉅亨新聞（code、time）：股票標記含這檔，或（news_title 為真時）標題提到簡稱。"""
    title = norm_text(news["title"].fillna(""))
    tags = news["stock"]
    out = []
    for r in rules.itertuples():
        hit = tags.map(lambda s: r.code in s).to_numpy(dtype=bool, copy=True)
        if r.news_title:
            hit |= mentions(title, r._replace(ptt_code=False, ptt_name=True))
        out.append(pd.DataFrame({"code": r.code, "time": news.loc[hit, "time"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


# ---- 有固定效果、雙向 cluster 的 OLS ----

def demean(df: pd.DataFrame, cols, fes, tol: float = 1e-10, max_iter: int = 10_000) -> pd.DataFrame:
    """交替投影：把 cols 對多組固定效果去平均（Frisch–Waugh–Lovell）。"""
    x = df[cols].to_numpy(dtype=float).copy()
    codes = [pd.factorize(df[f])[0] for f in fes]
    sizes = [np.bincount(k) for k in codes]
    for _ in range(max_iter):
        delta = 0.0
        for k, n in zip(codes, sizes):
            m = np.column_stack([np.bincount(k, weights=x[:, j]) / n for j in range(x.shape[1])])
            x -= m[k]
            delta = max(delta, np.abs(m).max())
        if delta < tol:
            break
    return pd.DataFrame(x, columns=cols, index=df.index)


def fe_ols(df: pd.DataFrame, y: str, xs, fes, clusters=("code", "date")):
    """y ~ xs + 固定效果，標準誤依 clusters 雙向 cluster（Cameron–Gelbach–Miller，statsmodels）。"""
    import statsmodels.api as sm
    d = df.dropna(subset=[y, *xs])
    dm = demean(d, [y, *xs], fes)
    groups = np.column_stack([pd.factorize(d[c])[0] for c in clusters])
    return sm.OLS(dm[y], dm[list(xs)]).fit(cov_type="cluster", cov_kwds={"groups": groups})


# ---- 分析 ----

def add_windows(d: pd.DataFrame, items: pd.DataFrame, days: pd.DatetimeIndex, prefix: str,
                start=None, end=None) -> pd.DataFrame:
    """d（days 表的列，要有 code、pos）加上五個窗口的則數與基準期，欄名 {prefix}_{窗口}、{prefix}_base。

    items：code、time。start／end：資料的起訖（end 不含）；窗口或基準期有一部分落在資料範圍外的是缺值。
    """
    parts = day_parts(items["time"], days)
    parts["code"] = items["code"].to_numpy()[parts.index]
    out = pd.DataFrame(index=d.index, columns=[f"{prefix}_{w}" for w in [*WINDOWS, "base"]], dtype=float)
    by_code = dict(tuple(parts.groupby("code")))
    empty = parts.iloc[:0]
    for c, rows in d.groupby("code"):
        pre, intra = daily_counts(by_code.get(c, empty), len(days))
        wc = window_counts(pre, intra, rows["pos"].to_numpy())
        out.loc[rows.index] = wc.to_numpy()
    i = d["pos"].to_numpy()
    if start is not None:   # 最早用到的是基準期第一天（t−25）那個窗口，從 t−26 收盤開始
        lo = np.clip(i - BASE[0] - 1, 0, None)
        out.loc[(i - BASE[0] - 1 < 0) | (days[lo] < pd.Timestamp(start))] = np.nan
    if end is not None:     # 最晚用到的是事後窗口的最後一天（t+5）
        hi = np.clip(i + 5, None, len(days) - 1)
        out.loc[(i + 5 >= len(days)) | (days[hi] >= pd.Timestamp(end))] = np.nan
    return d.join(out)


def outcomes(d: pd.DataFrame) -> pd.DataFrame:
    """則數 -> 結果變數：news_any_*（0/1）、news_log_*（log1p）、ptt_abn_*（異常討論量）。"""
    out = {}
    for w, n in WINDOWS.items():
        out[f"news_any_{w}"] = (d[f"news_{w}"] > 0).astype(float).where(d[f"news_{w}"].notna())
        out[f"news_log_{w}"] = np.log1p(d[f"news_{w}"])
        out[f"ptt_abn_{w}"] = np.log1p(d[f"ptt_{w}"] / n) - d["ptt_base"]
    return d.assign(**out)


def stock_lags(d: pd.DataFrame) -> pd.DataFrame:
    """同一檔前一個交易日的量價（P1、P2 的控制變數），以及下一個交易日的當沖比例。"""
    d = d.sort_values(["code", "date"])
    g = d.groupby("code", sort=False)
    lv = np.log(d["volume"])
    lv_avg = lv.groupby(d["code"]).transform(lambda s: s.shift(2).rolling(20).mean())
    return d.assign(
        abs_z1=(g["ar"].shift(1).abs() / d["sigma"]),
        log_sigma=np.log(d["sigma"]),
        lock1=g["locked"].shift(1).astype(float),
        lvol_abn1=lv.groupby(d["code"]).shift(1) - lv_avg,
        dt_next=g["dt_intraday"].shift(-1),
    )


def scar(d: pd.DataFrame, k: int, end) -> pd.Series:
    """sign(ar_t) × 之後 k 個交易日（這檔自己的）ar 的和；有任何一天缺值或落在 end 之後就是缺值。"""
    d = d.sort_values(["code", "date"])
    ar = d["ar"].where(d["date"] < pd.Timestamp(end))
    tot = sum(ar.groupby(d["code"]).shift(-j) for j in range(1, k + 1))
    return (np.sign(d["ar"]) * tot).reindex(d.index)


def part1(d: pd.DataFrame, ys) -> pd.DataFrame:
    """y = a × 大漲 + b × 大跌 + 個股×年 FE + 日期 FE；樣本是主要事件日加控制日。"""
    s = d[d["main"] | d["control"]].assign(up_ev=lambda x: (x["main"] & x["up"]).astype(float),
                                            dn_ev=lambda x: (x["main"] & ~x["up"]).astype(float))
    rows = []
    for y in ys:
        f = fe_ols(s, y, ["up_ev", "dn_ev"], ["code_year", "date"])
        ci = f.conf_int()
        raw = s.dropna(subset=[y])
        for x, lab in (("up_ev", "大漲"), ("dn_ev", "大跌")):
            rows.append({"y": y, "event": lab, "coef": f.params[x], "lo": ci.loc[x, 0], "hi": ci.loc[x, 1],
                         "p": f.pvalues[x], "n": int(f.nobs),
                         "mean_event": raw.loc[raw[x] == 1, y].mean(), "mean_control": raw.loc[raw["control"], y].mean()})
    return pd.DataFrame(rows)


P12_X = ["ptt_abn_preopen", "ptt_abn_pre5", "news_log_preopen", "news_log_pre5",
         "abs_z1", "log_sigma", "lock1", "lvol_abn1"]


def p12(d: pd.DataFrame):
    """事件(t) ~ 開盤前與事前的 PTT 異常 + 控制；樣本是所有合格、不是後續事件的日子。"""
    s = d[d["eligible"] & ~d["followon"]].assign(y=lambda x: x["main"].astype(float))
    return fe_ols(s, "y", P12_X, ["code_year", "date"])


P34_X = ["nonews", "locked", "abs_z", "log_sigma", "up_f"]


def p34(d: pd.DataFrame, k: int):
    """sCAR(k) ~ 無新聞 + 收盤鎖漲跌停 + |ar|/σ + log σ + 大漲 + 年月 FE；樣本是主要事件。"""
    s = d[d["main"]].assign(nonews=lambda x: ((x["news_preopen"] + x["news_intraday"]) == 0).astype(float)
                            .where(x["news_preopen"].notna() & x["news_intraday"].notna()),
                            locked=lambda x: x["locked"].astype(float), abs_z=lambda x: x["z"].abs(),
                            up_f=lambda x: x["up"].astype(float))
    return fe_ols(s, f"scar{k}", P34_X, ["year_month"])

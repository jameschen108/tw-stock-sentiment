"""用 Claude 標註一批 PTT 文字，當作訓練分類器的標籤（再用 classifier.py 蒸餾成小模型跑全量）。

- 大量標註走 Message Batches API（非同步、費用五折，通常一小時內完成）。
- sync 模式用來先試 10–20 則，確認提示詞與輸出正常。
- 刻意不給發文日期，並要求模型不要用文本以外的知識，降低「模型知道後來行情」造成的偷看。
需要：pip install anthropic，並設定 ANTHROPIC_API_KEY（或 `ant auth login`）。
"""
import json

import numpy as np
import pandas as pd

LABELS = ["bullish", "bearish", "neutral", "irrelevant"]

SYSTEM = """你是台灣股市社群文本的標註員。判斷一段 PTT Stock 板文字的作者，對「目標股票」未來股價有沒有自己的看法、看法是什麼。

標籤：
- bullish：作者自己認為目標股票會漲，或表達買進、加碼、續抱的意圖
- bearish：作者自己認為目標股票會跌，或表達賣出、停損、放空的意圖
- neutral：作者認真討論目標股票的前景（基本面、評價、籌碼、產業），但沒有方向，或多空並陳沒有結論
- irrelevant：沒有對目標股票未來股價的可用看法，包括：
  - 與目標股票無關
  - 只轉貼新聞、數據，沒有自己的評論
  - 幹話：玩梗、開玩笑、政治酸、酸別人或酸時事、炫耀或抱怨自己的損益、題外閒聊、單純發問

判斷順序：
1. 作者有沒有「自己」對目標股票未來走勢的看法？沒有 → irrelevant。
2. 有看法且有方向 → bullish / bearish。
3. 有認真討論但沒有方向或沒有結論 → neutral。
neutral 不是拿不準時的預設值。在 neutral 和 irrelevant 之間猶豫時，問自己：這段話能不能幫人判斷這檔股票接下來會漲還是跌、或提供了認真的分析？不能就標 irrelevant。
反過來，留言再短，只要作者對目標股票的基本面、需求、客戶、競爭對手、評價、股利或影響股價的因素提出自己的判斷，就算沒有方向也是 neutral，不是 irrelevant。

規則：
- 只根據文字本身判斷，不要使用你對這段期間實際行情的任何知識。
- 描述或抱怨已經發生、正在發生的漲跌或買賣（例如「今天又跌了」「外資就是要殺」「今天那麼弱外資當然賣」）不等於預測，標 irrelevant；作者接著說出未來方向（例如「還會再殺一波」）才標 bullish/bearish。
- sarcasm：PTT 常反串、反諷，例如「笑死 又要噴了」可能是嘲諷。是反諷就標 true，label 依作者真正的意思。
- 只有看得出「作者自己」的方向時才標 bullish/bearish。只是在嘲諷別人、名人、政府或時事，看不出作者自己的方向 → irrelevant；不要從被嘲諷的對象反推作者立場。
- 例外：拿公認的反指標開玩笑，而且明顯表示作者要反著做（例如「某某大師都喊買了，快逃」），可以標方向。
- banter：這段文字主要是幹話（上面列的類型）就標 true。幹話通常是 irrelevant；幹話裡若確實帶著作者自己的方向，可以標 bullish/bearish，banter 仍標 true。
- 文章若有「心得/評論」段，以作者的心得為準，不要拿新聞內文當作者的看法。
- evidence 請逐字引用支持判斷的原文片段（20 字以內），irrelevant 時留空字串。

示意（非本次資料）：
- 「這價位不買還等什麼，明天加碼」→ bullish
- 「毛利率指引下修，先出一半」→ bearish
- 「本益比在歷史區間中間，接下來看 AI 訂單能不能延續」→ neutral
- 「早知道十年前 all in，現在就不用上班了」→ irrelevant，banter
- 「某名嘴又要出來喊千元了，笑死」→ irrelevant，banter，sarcasm
- 「請問除息是哪天」→ irrelevant"""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "sarcasm": {"type": "boolean"},
        "banter": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["label", "sarcasm", "banter", "evidence"],
    "additionalProperties": False,
}

# 每百萬 token 的（輸入, 輸出, 快取讀取）美元；快取寫入是輸入的 1.25 倍（1 小時 TTL 為 2 倍），Batches API 再打五折
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}


def sample_texts(texts: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """依年度平均抽樣，留言約七成、文章約三成；去掉太短與重複的文字。"""
    df = texts[texts["text"].str.len() >= 6].drop_duplicates("text")
    df = df.assign(year=df["time"].dt.year)
    groups = df.groupby(["year", "kind"])
    per_year = n / df["year"].nunique()
    share = {"comment": 0.7, "article": 0.3}
    parts = [g.sample(min(len(g), int(round(per_year * share[k]))), random_state=seed)
             for (_, k), g in groups]
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["custom_id"] = [f"u{i:06d}" for i in range(len(out))]   # batch 的 id 只能用英數、-、_
    return out


def _user_message(row, ticker, name) -> str:
    text = row["text"][:2000]
    ctx = f"所在文章標題：{row['title']}\n" if row["kind"] == "comment" and row.get("title") else ""
    return f"目標股票：{ticker}（{name}）\n{ctx}{'留言' if row['kind'] == 'comment' else '文章'}內容：\n{text}"


def request_params(cfg, row, ticker, name, cache_ttl=None) -> dict:
    """cache_ttl="1h"：batch 會陸續處理幾十分鐘，預設 5 分鐘的快取容易過期，改用 1 小時提高命中率。"""
    cache_control = {"type": "ephemeral", **({"ttl": cache_ttl} if cache_ttl else {})}
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if cfg["llm"].get("effort"):
        output_config["effort"] = cfg["llm"]["effort"]
    return {
        "model": cfg["llm"]["model"],
        "max_tokens": 2048,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": cache_control}],
        "messages": [{"role": "user", "content": _user_message(row, ticker, name)}],
        "output_config": output_config,
    }


def _usage(msg) -> dict:
    """msg.model 是實際回答的模型（被拒答改用備援模型時會不同）。"""
    u = msg.usage
    return {"model_used": msg.model,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_write_tokens": u.cache_creation_input_tokens or 0,
            "cache_write_1h_tokens": getattr(u.cache_creation, "ephemeral_1h_input_tokens", 0) or 0,
            "cache_read_tokens": u.cache_read_input_tokens or 0}


def _parse_message(msg) -> dict:
    usage = _usage(msg)
    if msg.stop_reason == "refusal":
        return {"status": "refusal", **usage}
    if msg.stop_reason == "max_tokens":
        return {"status": "max_tokens", **usage}
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return {"status": "ok", **usage, **json.loads(text)}
    except json.JSONDecodeError:
        return {"status": "bad_json", **usage}


def _client():
    """.env 可設 ANTHROPIC_WORKSPACE_ID：API key 沒綁 workspace 時，請求要帶這個 header。"""
    import os

    import anthropic
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)


def submit_batch(cfg, sample: pd.DataFrame, ticker, name) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client()
    reqs = [Request(custom_id=r["custom_id"],
                    params=MessageCreateParamsNonStreaming(**request_params(cfg, r, ticker, name, "1h")))
            for _, r in sample.iterrows()]
    batch = client.messages.batches.create(requests=reqs)
    return batch.id


def batch_status(batch_id: str):
    return _client().messages.batches.retrieve(batch_id)


def collect_batch(batch_ids: list[str], sample: pd.DataFrame) -> pd.DataFrame:
    """合併多個 batch 的結果；只回傳有送出的那些則。"""
    client = _client()
    rows = []
    for batch_id in batch_ids:
        for res in client.messages.batches.results(batch_id):
            if res.result.type == "succeeded":
                rows.append({"custom_id": res.custom_id, **_parse_message(res.result.message)})
            else:   # errored / canceled / expired：可以重送
                rows.append({"custom_id": res.custom_id, "status": res.result.type})
    return sample.merge(pd.DataFrame(rows), on="custom_id", how="inner")


def label_sync(cfg, sample: pd.DataFrame, ticker, name) -> pd.DataFrame:
    """逐則呼叫，用來小量試跑。config 的 fallbacks 開啟時，被安全機制拒答會由伺服器自動改用其他模型。"""
    import anthropic
    client = _client()
    rows = []
    for _, r in sample.iterrows():
        params = request_params(cfg, r, ticker, name)
        try:
            if cfg["llm"].get("fallbacks"):
                msg = client.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"],
                                                  fallbacks="default")
            else:
                msg = client.messages.create(**params)
            rows.append({"custom_id": r["custom_id"], **_parse_message(msg)})
        except anthropic.RateLimitError:
            rows.append({"custom_id": r["custom_id"], "status": "rate_limited"})
        except anthropic.APIStatusError as e:
            rows.append({"custom_id": r["custom_id"], "status": f"http_{e.status_code}"})
        except anthropic.APIConnectionError:
            rows.append({"custom_id": r["custom_id"], "status": "connection_error"})
    return sample.merge(pd.DataFrame(rows), on="custom_id", how="left")


def to_training(labeled: pd.DataFrame) -> pd.DataFrame:
    """kind、title 是給 BERT 用的（留言以所在文章的標題當上下文）；TF-IDF 只看 text。"""
    ok = labeled[labeled["status"] == "ok"]
    return ok[["time", "text", "label", "kind", "title"]].assign(label=lambda d: d["label"].astype(str))


def label_counts(labeled: pd.DataFrame) -> dict:
    return {"status": labeled["status"].value_counts().to_dict(),
            "label": labeled.get("label", pd.Series(dtype=str)).value_counts().to_dict(),
            **{f"{c}_rate": float(np.nanmean(labeled.get(c, pd.Series([np.nan])).astype(float)))
               for c in ("sarcasm", "banter")}}


def usage_summary(labeled: pd.DataFrame, batch: bool = False) -> dict:
    """加總 token 並依 PRICES 估算美元；不在價目表裡的模型不計入 usd。"""
    cols = ["input_tokens", "output_tokens", "cache_write_tokens", "cache_write_1h_tokens",
            "cache_read_tokens"]
    if "model_used" not in labeled:
        return {}
    df = labeled.dropna(subset=["model_used"])
    if "cache_write_1h_tokens" not in df:   # 舊的結果沒有這欄
        df = df.assign(cache_write_1h_tokens=0)
    price = df["model_used"].map(PRICES)
    known = price.notna()
    inp = price[known].str[0] / 1e6
    out = price[known].str[1] / 1e6
    cache_read = price[known].str[2] / 1e6
    d = df[known]
    write_5m = d["cache_write_tokens"] - d["cache_write_1h_tokens"]
    usd = (d["input_tokens"] * inp + write_5m * inp * 1.25 + d["cache_write_1h_tokens"] * inp * 2
           + d["cache_read_tokens"] * cache_read + d["output_tokens"] * out).sum()
    return {**{c: int(df[c].sum()) for c in cols},
            "models": df["model_used"].value_counts().to_dict(),
            "usd": round(float(usd) * (0.5 if batch else 1.0), 4),
            "unpriced_models": sorted(set(df.loc[~known, "model_used"]))}

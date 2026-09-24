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

SYSTEM = """你是台灣股市社群文本的標註員。判斷一段 PTT Stock 板文字的作者，對「目標股票」未來股價的看法。

標籤：
- bullish：作者認為會漲、表達買進或續抱的意圖
- bearish：作者認為會跌、表達賣出、停損或放空的意圖
- neutral：有提到目標股票，但沒有方向，或多空並陳沒有結論
- irrelevant：與目標股票無關，或只是轉貼新聞、數據而沒有自己的看法

規則：
- 只根據文字本身判斷，不要使用你對這段期間實際行情的任何知識。
- 描述今天已經發生的漲跌（例如「今天又跌了」）不等於預測，除非作者接著表達未來方向。
- 注意 PTT 反串與反諷：「笑死 又要噴了」可能是嘲諷。sarcasm 標 true，label 依作者真正的意思。
- 嘲諷別人的看法，不代表作者持相反立場；看不出作者自己的方向就標 neutral。
- evidence 請逐字引用支持判斷的原文片段（20 字以內），irrelevant 時留空字串。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "sarcasm": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["label", "sarcasm", "evidence"],
    "additionalProperties": False,
}

# 每百萬 token 的（輸入, 輸出）美元；快取寫入 1.25 倍、讀取 0.1 倍，Batches API 再打五折
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
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


def request_params(cfg, row, ticker, name) -> dict:
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if cfg["llm"].get("effort"):
        output_config["effort"] = cfg["llm"]["effort"]
    return {
        "model": cfg["llm"]["model"],
        "max_tokens": 2048,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
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


def submit_batch(cfg, sample: pd.DataFrame, ticker, name) -> str:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    reqs = [Request(custom_id=r["custom_id"],
                    params=MessageCreateParamsNonStreaming(**request_params(cfg, r, ticker, name)))
            for _, r in sample.iterrows()]
    batch = client.messages.batches.create(requests=reqs)
    return batch.id


def batch_status(batch_id: str):
    import anthropic
    return anthropic.Anthropic().messages.batches.retrieve(batch_id)


def collect_batch(batch_id: str, sample: pd.DataFrame) -> pd.DataFrame:
    import anthropic
    client = anthropic.Anthropic()
    rows = []
    for res in client.messages.batches.results(batch_id):
        if res.result.type == "succeeded":
            rows.append({"custom_id": res.custom_id, **_parse_message(res.result.message)})
        else:   # errored / canceled / expired：可以重送
            rows.append({"custom_id": res.custom_id, "status": res.result.type})
    return sample.merge(pd.DataFrame(rows), on="custom_id", how="left")


def label_sync(cfg, sample: pd.DataFrame, ticker, name) -> pd.DataFrame:
    """逐則呼叫，用來小量試跑。claude-opus-5 預設開啟伺服器端 fallback（被安全機制拒答時自動改用其他模型）。"""
    import anthropic
    client = anthropic.Anthropic()
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
    ok = labeled[labeled["status"] == "ok"]
    return ok[["time", "text", "label"]].assign(label=lambda d: d["label"].astype(str))


def label_counts(labeled: pd.DataFrame) -> dict:
    return {"status": labeled["status"].value_counts().to_dict(),
            "label": labeled.get("label", pd.Series(dtype=str)).value_counts().to_dict(),
            "sarcasm_rate": float(np.nanmean(labeled.get("sarcasm", pd.Series([np.nan]))
                                             .astype(float)))}


def usage_summary(labeled: pd.DataFrame, batch: bool = False) -> dict:
    """加總 token 並依 PRICES 估算美元；不在價目表裡的模型不計入 usd。"""
    cols = ["input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens"]
    if "model_used" not in labeled:
        return {}
    df = labeled.dropna(subset=["model_used"])
    price = df["model_used"].map(PRICES)
    known = price.notna()
    inp = price[known].str[0] / 1e6
    out = price[known].str[1] / 1e6
    d = df[known]
    usd = (d["input_tokens"] * inp + d["cache_write_tokens"] * inp * 1.25
           + d["cache_read_tokens"] * inp * 0.1 + d["output_tokens"] * out).sum()
    return {**{c: int(df[c].sum()) for c in cols},
            "models": df["model_used"].value_counts().to_dict(),
            "usd": round(float(usd) * (0.5 if batch else 1.0), 4),
            "unpriced_models": sorted(set(df.loc[~known, "model_used"]))}

"""中文預訓練模型（預設 ckiplab/bert-base-chinese，繁體中文）微調的情緒分類器。

標籤與 classifier.py 相同（bullish / bearish / neutral / irrelevant），分數也相同：
P(看多) - P(看空)；判定為 irrelevant 時為 NaN。
輸入：留言把所在文章的標題當第一段、留言當第二段（LLM 標註時也看得到標題）；
太長的文字保留開頭 HEAD 個 token 加結尾，因為文章的作者心得通常在後面。
需要：pip install torch transformers。第一次使用會從 Hugging Face 下載模型（約 400 MB）。
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE_MODEL = "ckiplab/bert-base-chinese"
TOKENIZER = "bert-base-chinese"   # ckiplab 的模型沿用 Google bert-base-chinese 的詞表
LABELS = ["bearish", "bullish", "neutral", "irrelevant"]
MAX_LEN = 256
HEAD = 64
MAX_TITLE = 48


def _torch():
    import torch
    return torch


def device():
    torch = _torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(tokenizer, texts, titles=None, max_len=MAX_LEN, head=HEAD):
    """回傳（input_ids, token_type_ids）兩個 list。titles 為 None 或某則是空值時只用單段。"""
    texts = [str(t) for t in texts]
    titles = [None] * len(texts) if titles is None else [t if isinstance(t, str) and t else None for t in titles]
    body = tokenizer(texts, add_special_tokens=False)["input_ids"]
    head_ids = tokenizer([t or "" for t in titles], add_special_tokens=False)["input_ids"]
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    ids, types = [], []
    for b, ti, has_title in zip(body, head_ids, titles):
        ti = ti[:MAX_TITLE] if has_title else []
        room = max_len - 2 - (len(ti) + 1 if ti else 0)
        if len(b) > room:
            b = b[:head] + b[len(b) - (room - head):]
        first = [cls] + (ti + [sep] if ti else [])
        ids.append(first + b + [sep])
        types.append([0] * len(first) + [1 if ti else 0] * (len(b) + 1))
    return ids, types


def _batches(enc, pad_id, batch_size, order):
    torch = _torch()
    ids, types = enc
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        n = max(len(ids[j]) for j in idx)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        yield idx, {"input_ids": torch.tensor([pad(ids[j], pad_id) for j in idx]),
                    "token_type_ids": torch.tensor([pad(types[j], 0) for j in idx]),
                    "attention_mask": torch.tensor([[1] * len(ids[j]) + [0] * (n - len(ids[j])) for j in idx])}


def _length_batches(lengths, batch_size, rng):
    """長度相近的放同一批（padding 少、記憶體省很多），批次順序再隨機打亂。"""
    order = np.argsort(np.asarray(lengths) + rng.random(len(lengths)))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    return np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])


def predict_proba(model, tokenizer, texts, titles=None, batch_size=64) -> np.ndarray:
    """依長度排序後分批推論（padding 少、快很多），再還原原本順序。"""
    torch = _torch()
    enc = encode(tokenizer, texts, titles)
    dev = next(model.parameters()).device
    order = np.argsort([len(x) for x in enc[0]])
    out = np.zeros((len(enc[0]), model.config.num_labels), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for idx, batch in _batches(enc, tokenizer.pad_token_id, batch_size, order):
            logits = model(**{k: v.to(dev) for k, v in batch.items()}).logits
            out[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def predict(model, tokenizer, texts, titles=None, **kw) -> np.ndarray:
    return np.array(LABELS)[predict_proba(model, tokenizer, texts, titles, **kw).argmax(axis=1)]


def titles_of(df):
    """只有留言用標題當上下文；文章的 text 本身已經包含標題。"""
    return df["title"].where(df["kind"] == "comment") if "title" in df else None


def fine_tune(train: pd.DataFrame, valid: pd.DataFrame | None = None, epochs: int = 6,
              lr: float = 3e-5, batch_size: int = 16, seed: int = 0, log=print):
    """train / valid 需要欄位 text、label、kind、title。類別權重和 classifier.py 的 class_weight="balanced" 相同。

    有 valid 時每個 epoch 算一次 macro-F1，回傳最好那個 epoch 的模型；
    回傳（model, tokenizer, 每個 epoch 的 valid macro-F1 list，最好的 epoch 數）。
    """
    import copy

    torch = _torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(dev)

    enc = encode(tokenizer, train["text"], titles_of(train))
    y = torch.tensor(train["label"].map(LABELS.index).to_numpy())
    counts = np.bincount(y.numpy(), minlength=len(LABELS))
    weight = torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(y) / batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    history, best, best_state = [], -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        order = _length_batches([len(x) for x in enc[0]], batch_size, rng)
        for idx, batch in _batches(enc, tokenizer.pad_token_id, batch_size, order):
            logits = model(**{k: v.to(dev) for k, v in batch.items()}).logits
            loss = loss_fn(logits, y[idx].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(idx)
        msg = f"epoch {ep}  train loss {total / len(y):.4f}"
        if valid is not None:
            f1 = f1_score(valid["label"], predict(model, tokenizer, valid["text"], titles_of(valid)),
                          labels=LABELS, average="macro")
            history.append(f1)
            msg += f"  valid macro-F1 {f1:.3f}"
            if f1 > best:
                best, best_state = f1, copy.deepcopy(model.state_dict())
        log(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    best_epoch = int(np.argmax(history)) + 1 if history else epochs
    return model, tokenizer, history, best_epoch


def score(model, tokenizer, texts, titles=None, **kw) -> np.ndarray:
    proba = predict_proba(model, tokenizer, texts, titles, **kw)
    s = proba[:, LABELS.index("bullish")] - proba[:, LABELS.index("bearish")]
    return np.where(proba.argmax(axis=1) == LABELS.index("irrelevant"), np.nan, s)


def save(model, tokenizer, path: Path, meta: dict):
    import sklearn
    import torch
    import transformers
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    meta = {**meta, "base_model": BASE_MODEL, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__, "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now().isoformat(timespec="seconds")}
    (path / "training.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                        encoding="utf-8")


def load(path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先跑 scripts/train_classifier.py --labels llm_pooled --model bert")
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device())
    return model, AutoTokenizer.from_pretrained(path)


def load_meta(path: Path) -> dict:
    p = path / "training.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

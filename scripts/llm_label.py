"""用 Claude 標註情緒（選用，會產生 API 費用）。先 pip install anthropic 並設定 ANTHROPIC_API_KEY。

    python scripts/llm_label.py sample            # 抽樣（不花錢）
    python scripts/llm_label.py sync --n 20       # 先試 20 則，檢查標得合不合理
    python scripts/llm_label.py submit            # 整批送 Batches API（五折，通常一小時內完成）
    python scripts/llm_label.py status
    python scripts/llm_label.py collect           # 下載結果 -> data/llm/labels_{ticker}.parquet
之後：python scripts/train_classifier.py --labels llm
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import load_config, work_path  # noqa: E402
from pttsent.ptt import ticker_aliases  # noqa: E402
from pttsent.sentiment import llm_label as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "sync", "submit", "status", "collect"])
    ap.add_argument("--ticker")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config()
    ticker = a.ticker or str(cfg["ticker"])
    name = ticker_aliases(cfg, ticker)[0]
    sample_path = work_path(cfg, "llm", f"sample_{ticker}.parquet")
    batch_path = work_path(cfg, "llm", f"batch_{ticker}.json")

    if a.cmd == "sample":
        texts = pd.read_parquet(work_path(cfg, "interim", f"texts_{ticker}.parquet"))
        texts = texts[texts["time"].dt.year.isin(cfg["ptt_years"])]
        s = L.sample_texts(texts, cfg["llm"]["n_samples"], cfg["llm"]["seed"])
        s.to_parquet(sample_path, index=False)
        print(f"抽出 {len(s):,} 則 {s.groupby(['year', 'kind']).size().unstack().to_dict()}")
        print(f"-> {sample_path}")
        return

    sample = pd.read_parquet(sample_path)
    if a.cmd == "sync":
        res = L.label_sync(cfg, sample.head(a.n), ticker, name)
        res.to_parquet(work_path(cfg, "llm", f"sync_{ticker}.parquet"), index=False)
        print(L.label_counts(res))
        print("usage:", L.usage_summary(res))
        for _, r in res.iterrows():
            print(f"[{r.get('label')}{' 反諷' if r.get('sarcasm') is True else ''}{' 幹話' if r.get('banter') is True else ''}] "
                  f"{r['text'][:60]!r}  <- {r.get('evidence')!r}")
    elif a.cmd == "submit":
        batch_id = L.submit_batch(cfg, sample, ticker, name)
        json.dump({"batch_id": batch_id, "n": len(sample), "model": cfg["llm"]["model"],
                   "created": datetime.now().isoformat()}, open(batch_path, "w"), indent=2)
        print(f"已送出 batch {batch_id}（{len(sample):,} 則）-> {batch_path}")
    elif a.cmd == "status":
        b = L.batch_status(json.load(open(batch_path))["batch_id"])
        print(b.processing_status, b.request_counts)
    elif a.cmd == "collect":
        res = L.collect_batch(json.load(open(batch_path))["batch_id"], sample)
        out = work_path(cfg, "llm", f"labels_{ticker}.parquet")
        res.to_parquet(out, index=False)
        print(L.label_counts(res))
        print("usage:", L.usage_summary(res, batch=True))
        print(f"-> {out}")


if __name__ == "__main__":
    main()

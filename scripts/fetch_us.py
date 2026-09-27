"""下載台積電 ADR（TSM）與費城半導體指數（^SOX）的日資料（Yahoo Finance chart API），給 11_volume_prereg.py 用。

    python scripts/fetch_us.py            # 已經有的檔案會跳過
    python scripts/fetch_us.py --force    # 重新下載
    python scripts/fetch_us.py --end 2026-09-27 --out us_2026   # 延長期間另存，不動 2024 登記用的檔案

存到 data/prices/{--out，預設 us}/{TSM,SOX}.json，旁邊的 .meta.json 記錄網址、抓取時間、筆數與 sha256。
報酬用 adjclose（含息還原）；之後若再配息，Yahoo 會把之前的 adjclose 等比例調整，對數報酬不變。
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import load_config, work_path  # noqa: E402

SYMBOLS = {"TSM": "TSM", "SOX": "^SOX"}
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
       "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
HEADERS = {"User-Agent": "Mozilla/5.0 (research; PTT sentiment thesis)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2025-07-01")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="us", help="data/prices/ 底下的資料夾")
    a = ap.parse_args()
    cfg = load_config()
    p1, p2 = (int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()) for s in (a.start, a.end))
    for name, sym in SYMBOLS.items():
        out = work_path(cfg, "prices", a.out, f"{name}.json")
        if out.exists() and not a.force:
            print(f"{name}：已存在，跳過（--force 重新下載）")
            continue
        url = URL.format(sym=urllib.parse.quote(sym), p1=p1, p2=p2)
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
            raw = r.read()
        res = json.loads(raw)["chart"]["result"][0]
        out.write_bytes(raw)
        meta = {"source": "Yahoo Finance chart API", "symbol": sym, "url": url,
                "fetched_on": datetime.now().astimezone().isoformat(), "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "n_rows": len(res["timestamp"]),
                "exchange_tz": res["meta"].get("exchangeTimezoneName")}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"{name}：{meta['n_rows']} 天 -> {out}")
        time.sleep(2)


if __name__ == "__main__":
    main()

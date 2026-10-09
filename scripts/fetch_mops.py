"""下載 267 檔的 MOPS 重大訊息（舊版網站 t05st01，依公司、年度查詢）。

    python scripts/fetch_mops.py                          # 2019–2023
    python scripts/fetch_mops.py --start 2019 --end 2023 --codes 2330 1616
    python scripts/fetch_mops.py --start 2026 --end 2027 --out t05st01_2027   # 另存一個資料夾（大漲跌預先登記的評估期 B）

每檔每年一個請求，存到 data/mops/t05st01/{代號}_{民國年}.html；一年接近 200 則時（怕被截斷）再逐月各查一次，
存成 {代號}_{民國年}_{月}.html。manifest.jsonl 記錄每個請求的參數、時間、狀態、則數與 sha256；中斷後重跑會跳過
已經記錄的請求。請求間隔預設 4 秒：MOPS 會封鎖太頻繁的查詢。共用資料夾（PTT_DATA_DIR）不會被寫入。
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent import mops  # noqa: E402
from pttsent.config import load_config, work_path  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (research; PTT sentiment thesis)",
           "Content-Type": "application/x-www-form-urlencoded"}


def fetch(data: dict) -> str:
    req = urllib.request.Request(mops.URL, data=urllib.parse.urlencode(data).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--codes", nargs="*", help="預設是 universe_267.csv 全部")
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--out", default="t05st01", help="data/mops/ 底下的資料夾")
    a = ap.parse_args()
    cfg = load_config()
    out_dir = work_path(cfg, "mops", a.out, "x").parent
    manifest = out_dir / "manifest.jsonl"
    log = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()] if manifest.exists() else []
    done = {r["key"] for r in log}

    codes = a.codes or pd.read_csv(Path(cfg["data_dir"]) / "universe_267.csv", dtype=str,
                                   encoding="utf-8-sig")["ticker"].tolist()
    jobs = [(c, y - 1911, None) for c in codes for y in range(a.start, a.end + 1)]
    todo = [j for j in jobs if f"{j[0]}_{j[1]}" not in done]
    # 中斷在逐月查詢中間的年度：補上還沒查的月份
    todo += [(r["code"], r["roc_year"], m) for r in log if r["month"] is None and r["n"] >= mops.SPLIT_AT
             for m in range(1, 13) if f"{r['code']}_{r['roc_year']}_{m:02d}" not in done]
    print(f"要查 {len(todo)} 個公司年度（已完成 {len(jobs) - len(todo)}），預估 {len(todo) * a.sleep / 3600:.1f} 小時", flush=True)

    fails = 0
    while todo:
        code, roc, month = todo.pop(0)
        key = f"{code}_{roc}" + (f"_{month:02d}" if month else "")
        try:
            page = fetch(mops.form(code, roc, month))
            st = mops.status(page)
            if st == "blocked":
                raise RuntimeError("回應不是結果頁（可能查詢太頻繁）")
        except Exception as e:     # 被擋或連線錯誤：等久一點再試，連續失敗就停
            fails += 1
            print(f"{key} 失敗（第 {fails} 次）：{type(e).__name__} {str(e)[:80]}", flush=True)
            if fails >= 5:
                sys.exit("連續失敗 5 次，可能被暫時封鎖；稍後重跑會從這裡接續")
            todo.insert(0, (code, roc, month))
            time.sleep(60 * fails)
            continue
        fails = 0
        n = len(mops.parse(page)) if st == "ok" else 0
        (out_dir / f"{key}.html").write_text(page, encoding="utf-8")
        if month is None and n >= mops.SPLIT_AT:
            todo[:0] = [(code, roc, m) for m in range(1, 13)]
        with manifest.open("a") as f:
            f.write(json.dumps({"key": key, "code": code, "roc_year": roc, "month": month, "status": st, "n": n,
                                "sha256": hashlib.sha256(page.encode()).hexdigest(),
                                "fetched_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
        print(f"{key} {st} {n}", flush=True)
        time.sleep(a.sleep)


if __name__ == "__main__":
    main()

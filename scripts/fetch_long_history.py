"""下載研究五（槓桿 ETF）要用的長歷史：台指期、加權指數、加權報酬指數、央行利率。

    python scripts/fetch_long_history.py              # 已經有的檔案會跳過；全部約 35 分鐘（證交所每次間隔 3 秒）
    python scripts/fetch_long_history.py --only tx    # 只抓一種：tx | tr | price | cbc

存到 data/long/：
- tx/TaiwanFuturesDaily_TX_{年}.json：FinMind TaiwanFuturesDaily，1998-07（台指期上市）起。
  期交所官方 futDataDown 在 2026-10 測試時一律回「日期時間錯誤」，所以用 FinMind；
  2014 年以後可以和共用資料夾 raw/taifex_tx/ 的官方檔核對（etf.compare_tx_official）。
- tr/MFI94U_{年月}.json：證交所發行量加權股價報酬指數（含息），2003-01（指數起始）起。
- price/MI_5MINS_HIST_{年月}.json：證交所發行量加權股價指數，1999-01 起，只當描述。
- cbc/EG28D01.csv：中央銀行重貼現率（政府資料開放平臺 dataset 10783），只有調整日。
每個檔旁邊的 .meta.json 記錄網址、抓取時間、大小與 sha256。共用資料夾不會被寫入。
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pttsent.config import load_config, work_path  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (research; PTT sentiment thesis)"}
TX = "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanFuturesDaily&data_id=TX&start_date={s}&end_date={e}"
TR = "https://www.twse.com.tw/rwd/zh/TAIEX/MFI94U?date={ym}01&response=json"
PRICE = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={ym}01"
CBC = "https://www.cbc.gov.tw/public/data/OpenData/" + urllib.parse.quote("經研處") + "/EG28D01.csv"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
        return r.read()


def save(out: Path, raw: bytes, url: str, **extra):
    out.write_bytes(raw)
    meta = {"url": url, "fetched_on": datetime.now().astimezone().isoformat(), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), **extra}
    out.with_name(out.name + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def get_tx(cfg, end: date, force: bool):
    for y in range(1998, end.year + 1):
        out = work_path(cfg, "long", "tx", f"TaiwanFuturesDaily_TX_{y}.json")
        if out.exists() and not force:
            continue
        url = TX.format(s=f"{y}-01-01", e=min(date(y, 12, 31), end).isoformat())
        raw = fetch(url)
        js = json.loads(raw)
        if js.get("status") != 200 or not js.get("data"):
            print(f"tx {y}：失敗（{js.get('msg')}），下次再抓", flush=True)
        else:
            save(out, raw, url, source="FinMind TaiwanFuturesDaily", n_rows=len(js["data"]))
            print(f"tx {y}：{len(js['data'])} 筆", flush=True)
        time.sleep(3)


def get_twse(cfg, kind: str, start: date, end: date, force: bool):
    url_fmt, prefix = (TR, "MFI94U") if kind == "tr" else (PRICE, "MI_5MINS_HIST")
    for ym in months(start, end):
        out = work_path(cfg, "long", kind, f"{prefix}_{ym}.json")
        if out.exists() and not force:
            continue
        url = url_fmt.format(ym=ym)
        try:
            raw = fetch(url)
            js = json.loads(raw)
        except Exception as e:      # 一次失敗就記下來，下次再抓，不中斷
            print(f"{kind} {ym}：失敗（{type(e).__name__}）", flush=True)
            time.sleep(10)
            continue
        if js.get("stat") != "OK":
            print(f"{kind} {ym}：{js.get('stat')}", flush=True)
        else:
            save(out, raw, url, source=f"TWSE {prefix}", n_rows=len(js.get("data") or []))
            print(f"{kind} {ym}：{len(js.get('data') or [])} 天", flush=True)
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--only", choices=["tx", "tr", "price", "cbc"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    end = date.fromisoformat(a.end)
    if a.only in (None, "cbc"):
        out = work_path(cfg, "long", "cbc", "EG28D01.csv")
        if not out.exists() or a.force:
            save(out, fetch(CBC), CBC, source="中央銀行利率（data.gov.tw dataset 10783）")
            print("cbc：完成", flush=True)
    if a.only in (None, "tx"):
        get_tx(cfg, end, a.force)
    if a.only in (None, "tr"):
        get_twse(cfg, "tr", date(2003, 1, 1), end, a.force)
    if a.only in (None, "price"):
        get_twse(cfg, "price", date(1999, 1, 1), end, a.force)


if __name__ == "__main__":
    main()

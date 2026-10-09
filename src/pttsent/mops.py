"""公開資訊觀測站（MOPS）重大訊息：舊版網站 mopsov 的 t05st01（依公司、年度查詢），由 scripts/fetch_mops.py 下載。"""
import html
import re
from pathlib import Path

import pandas as pd

URL = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"
NO_DATA = "查無需求資料"
SPLIT_AT = 190      # 一年的則數接近 200 時，怕被截斷，改成逐月再查一次

ROW_RE = re.compile(r"<tr class='(?:even|odd)'>(.*?)</tr>", re.S)
CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
SEQ_RE = re.compile(r"seq_no\.value='(\d+)'")


def form(code: str, roc_year: int, month: int | None = None) -> dict:
    d = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "queryName": "co_id", "inpuType": "co_id",
         "TYPEK": "all", "co_id": code, "year": roc_year}
    if month is not None:
        d["month"] = f"{month:02d}"
    return d


def status(page: str) -> str:
    """ok：有結果表；empty：查無資料；blocked：其他（通常是查詢太頻繁被擋）。"""
    if "class='hasBorder'" in page:
        return "ok"
    if NO_DATA in page:
        return "empty"
    return "blocked"


def _clean(cell: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", cell))).strip()


def parse(page: str) -> pd.DataFrame:
    """結果頁 -> 每則一列：code、time（台北時間）、seq_no、subject。"""
    rows = []
    for tr in ROW_RE.findall(page):
        cells = [_clean(c) for c in CELL_RE.findall(tr)]
        if len(cells) < 5 or not re.fullmatch(r"\d{2,3}/\d{2}/\d{2}", cells[2]):
            continue
        y, m, d = map(int, cells[2].split("/"))
        seq = SEQ_RE.search(tr)
        rows.append((cells[0], pd.Timestamp(f"{y + 1911}-{m:02d}-{d:02d} {cells[3]}"),
                     int(seq.group(1)) if seq else None, cells[4]))
    d = pd.DataFrame(rows, columns=["code", "time", "seq_no", "subject"])
    d["time"] = pd.to_datetime(d["time"])
    return d


def load(out_dir: Path) -> pd.DataFrame:
    """讀所有下載的頁面；有逐月檔的年度用逐月檔，同一則（code、time、seq_no）只留一次。"""
    files = sorted(out_dir.glob("*.html"))
    monthly = {f.stem.rsplit("_", 1)[0] for f in files if f.stem.count("_") == 2}
    use = [f for f in files if f.stem.count("_") == 2 or f.stem not in monthly]
    d = pd.concat([parse(f.read_text(encoding="utf-8", errors="replace")) for f in use], ignore_index=True)
    return d.drop_duplicates(["code", "time", "seq_no", "subject"]).sort_values(["code", "time"]).reset_index(drop=True)

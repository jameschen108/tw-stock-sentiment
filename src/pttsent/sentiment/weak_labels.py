"""從 [標的] 文章的「分類：多／空」抽弱標籤。

板規要求 [標的] 文填分類，等於作者自己標了多空。要注意：
- 分類那行和標題（格式含多空）都要從文字裡拿掉，否則模型只是在學「分類：多」四個字。
- 看多遠多於看空（2023 年約 9:1），訓練時要處理類別不平衡。
- 這是文章層級的標籤，拿去預測短留言會有落差。
"""
import re
from pathlib import Path

import pandas as pd

LABEL_RE = re.compile(r"分類\s*[:：]\s*([^\n]{0,8})")


def label_from_content(content: str):
    m = LABEL_RE.search(content or "")
    if not m:
        return None
    v = m.group(1)
    bull, bear = "多" in v, "空" in v
    if bull and not bear:
        return "bullish"
    if bear and not bull:
        return "bearish"
    return None   # 「多/空/」這種沒刪範本的，或「討論」「請益」


def training_text(content: str) -> str:
    return "\n".join(ln for ln in (content or "").splitlines() if "分類" not in ln).strip()


def weak_label_dataset(ptt_dir: Path, years) -> pd.DataFrame:
    rows = []
    for y in years:
        a = pd.read_parquet(ptt_dir / f"articles_{y}.parquet",
                            columns=["article_id", "time", "category", "is_reply", "content"])
        a = a[(a["category"] == "標的") & ~a["is_reply"]]
        a = a.assign(label=a["content"].map(label_from_content)).dropna(subset=["label"])
        a["text"] = a["content"].map(training_text)
        rows.append(a[a["text"].str.len() >= 20][["article_id", "time", "text", "label"]])
    return pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)

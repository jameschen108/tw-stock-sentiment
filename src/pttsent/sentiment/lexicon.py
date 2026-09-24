"""PTT 股票板多空詞典（基準方法）。

分數 = (看多詞數 - 看空詞數) / (看多詞數 + 看空詞數)，沒有命中任何詞 -> NaN。
已知限制：分不出描述過去（「今天崩了」）和預測未來（「明天會崩」），也抓不到反串。
這是用來當基準的，之後應該拿 LLM 標註或弱標籤訓練的分類器比較它。
"""
import re
import unicodedata

import numpy as np

BULL = """
噴 噴出 大噴 狂噴 起飛 飛天 漲停 鎖漲停 漲爆 大漲 狂漲 創新高 新高 突破 長紅 紅k
看多 做多 偏多 多軍 多頭 看好 利多 看漲 會漲 必漲 起漲 補漲 反彈 跌深反彈 軋空
加碼 進場 上車 抄底 撿便宜 低接 買進 買入 買爆 買起來 歐印 all in 梭哈 續抱 抱緊
發大財 賺爆 穩了 護盤 上看
""".split()

BEAR = """
崩 崩盤 大跌 重挫 暴跌 跌停 鎖跌停 下殺 殺盤 破底 跌破 新低 長黑 黑k
看空 做空 偏空 空軍 空頭 看壞 利空 利多出盡 看跌 會跌 跌爆 泡沫 見頂 頭部
出貨 倒貨 逃命 快逃 快跑 出清 停損 認賠 套牢 被套 住套房 套房 韭菜 割韭菜 畢業
放空 下車 賣出 賣光 完蛋 住海景
""".split()

# 整則留言只有一個字時才算（「多」「空」單字在一般句子裡太常見）
SOLO = {"多": 1, "噴": 1, "漲": 1, "空": -1, "崩": -1, "跌": -1}
NEGATORS = set("不沒別未無")

_POLARITY = {w: 1 for w in BULL} | {w: -1 for w in BEAR}
# 長詞優先，「利多出盡」要比「利多」先匹配
_TERM_RE = re.compile("|".join(re.escape(w) for w in sorted(_POLARITY, key=len, reverse=True)))
_STRIP_RE = re.compile(r"[\W_]+")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def score_text(text: str) -> float:
    t = _normalize(text)
    core = _STRIP_RE.sub("", t)
    if core in SOLO:
        return float(SOLO[core])
    pos = neg = 0
    for m in _TERM_RE.finditer(t):
        p = _POLARITY[m.group(0)]
        if NEGATORS & set(t[max(0, m.start() - 3):m.start()]):
            p = -p   # 「不會崩」「不看好」
        if p > 0:
            pos += 1
        else:
            neg += 1
    if pos + neg == 0:
        return np.nan
    return (pos - neg) / (pos + neg)


def score_texts(texts) -> np.ndarray:
    return np.array([score_text(t) for t in texts], dtype=float)

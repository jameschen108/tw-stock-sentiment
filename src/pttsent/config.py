"""讀取 config.yaml，集中管理路徑。"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["lexicon", "classifier_weak", "classifier_llm"]
TARGETS = ["close_to_close", "open_to_close"]


def target_suffix(target: str) -> str:
    """close_to_close 沿用原本的檔名；其他目標加後綴，避免蓋掉彼此的結果。"""
    return "" if target == "close_to_close" else "_oc"


def load_env(path=None) -> None:
    """讀專案根目錄的 .env（KEY=VALUE 一行一個），已存在的環境變數優先。"""
    path = Path(path or ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def load_config(path=None) -> dict:
    load_env()
    path = Path(path or os.environ.get("PTTSENT_CONFIG", ROOT / "config.yaml"))
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data_dir"] = Path(os.environ.get("PTT_DATA_DIR", cfg["data_dir"]))
    for key in ("work_dir", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else ROOT / p
    p = Path(cfg["sentiment"]["classifier_dir"])
    cfg["sentiment"]["classifier_dir"] = p if p.is_absolute() else ROOT / p
    return cfg


def classifier_path(cfg, labels: str, ticker: str) -> Path:
    """弱標籤不分股票，共用一個模型；LLM 標籤是針對某檔股票標的，每檔各一個。"""
    name = "sentiment_clf_weak" if labels == "weak" else f"sentiment_clf_llm_{ticker}"
    return cfg["sentiment"]["classifier_dir"] / f"{name}.joblib"


def work_path(cfg, *parts) -> Path:
    p = cfg["work_dir"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def output_path(cfg, *parts) -> Path:
    p = cfg["output_dir"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

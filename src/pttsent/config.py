"""讀取 config.yaml，集中管理路徑。"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    path = Path(path or os.environ.get("PTTSENT_CONFIG", ROOT / "config.yaml"))
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data_dir"] = Path(os.environ.get("PTT_DATA_DIR", cfg["data_dir"]))
    for key in ("work_dir", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else ROOT / p
    p = Path(cfg["sentiment"]["classifier_path"])
    cfg["sentiment"]["classifier_path"] = p if p.is_absolute() else ROOT / p
    return cfg


def work_path(cfg, *parts) -> Path:
    p = cfg["work_dir"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def output_path(cfg, *parts) -> Path:
    p = cfg["output_dir"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

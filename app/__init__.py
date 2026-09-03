"""从 cfg/*.yaml 加载配置并导出页面可选项常量。"""

from pathlib import Path

import yaml

CFG_DIR = Path(__file__).parent.parent / "cfg"


def _load(name: str) -> dict:
    """读取 cfg/<name>.yaml 并返回解析后的字典。"""
    with open(CFG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


_models = _load("models")
_inference = _load("inference")

ALGORITHMS = _models["algorithms"]
MODELS = _models["models"]
METRICS = _models["metrics"]

RESULT_DIR = _inference["result_dir"]
CHECKPOINT_DIR = _inference["checkpoint_dir"]

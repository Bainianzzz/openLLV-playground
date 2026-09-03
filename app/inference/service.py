from pathlib import Path

import openLLV as llv

from app import RESULT_DIR
from app.inference.validate import _parse_params, _source

DEFAULT_RESULT_DIR = Path(RESULT_DIR)


def _run(method, image, input_dir, save_dir, kwargs):
    """执行预测并返回 (结果图, 状态)。

    文件夹输入走批量推理（不返回单图）；单图返回增强后的 numpy 数组。
    """
    source = _source(image, input_dir)
    out_dir = Path(save_dir.strip()) if save_dir and save_dir.strip() else DEFAULT_RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, str) and Path(source).is_dir():
        llv.predict(method, source, output=str(out_dir), **kwargs)
        return None, "finish"

    out = out_dir / "output.png"
    _, saved = llv.predict(method, source, output=str(out), **kwargs)
    return llv.imread(saved, output_format="numpy"), "finish"


def run_tradition_inference(image, input_dir, algo, save_dir, params):
    """执行传统算法推理，返回增强结果与状态。"""
    return _run(algo, image, input_dir, save_dir, _parse_params(params))


def run_deep_inference(image, input_dir, checkpoint, device, resize, save_dir):
    """执行深度模型推理（需 checkpoint），返回增强结果与状态。"""
    kwargs = {"device": device}
    if resize:
        kwargs["resize"] = int(resize)
    return _run(checkpoint, image, input_dir, save_dir, kwargs)

"""推理输入的参数解析与来源校验。"""

import json

import gradio as gr


def _parse_params(raw):
    """解析算法参数字符串为字典；空输入返回空字典，非 JSON 或非对象抛 gr.Error。"""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise gr.Error("算法参数必须是合法的 JSON，例如 {\"gamma\": 0.8}")
    if not isinstance(parsed, dict):
        raise gr.Error("算法参数必须是 JSON 对象")
    return parsed


def _source(image, input_dir):
    """解析推理输入来源：单张图像或文件夹，二者互斥。"""
    if image is not None and (input_dir and input_dir.strip()):
        raise gr.Error("单张图像与文件夹输入不能同时使用，二选一")
    if image is not None:
        return image
    if input_dir and input_dir.strip():
        return input_dir.strip()
    raise gr.Error("请先上传图像或指定输入文件夹")

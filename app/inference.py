import json
from pathlib import Path

import gradio as gr
import openLLV as llv

from app import ALGORITHMS, RESULT_DIR

DEFAULT_RESULT_DIR = Path(RESULT_DIR)


def _parse_params(raw):
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise gr.Error("算法参数必须是合法的 JSON，例如 {\"gamma\": 0.8}")
    if not isinstance(parsed, dict):
        raise gr.Error("算法参数必须是 JSON 对象")
    return parsed


def run_tradition_inference(image, algo, save_dir, params):
    if image is None:
        raise gr.Error("请先上传图像")
    kwargs = _parse_params(params)
    out_dir = Path(save_dir.strip()) if save_dir and save_dir.strip() else DEFAULT_RESULT_DIR
    out = out_dir / "output.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _, saved = llv.predict(algo, image, output=str(out), **kwargs)
    return llv.imread(saved, output_format="numpy"), str(saved)


def run_deep_inference(image, checkpoint, device, resize, save_dir):
    if image is None:
        raise gr.Error("请先上传图像")
    out_dir = Path(save_dir.strip()) if save_dir and save_dir.strip() else DEFAULT_RESULT_DIR
    out = out_dir / "output.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"device": device}
    if resize:
        kwargs["resize"] = int(resize)
    _, saved = llv.predict(checkpoint, image, output=str(out), **kwargs)
    return llv.imread(saved, output_format="numpy"), str(saved)


def build():
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="输入图像")
        with gr.Column(scale=1):
            output_img = gr.Image(label="结果")
            output_path = gr.Textbox(label="保存路径")

    with gr.Tabs():
        with gr.Tab("传统增强"):
            algo = gr.Dropdown(ALGORITHMS, value=ALGORITHMS[0], label="算法")
            traditional_save_dir = gr.Textbox(
                label="结果保存目录（留空默认 result/）",
                value=RESULT_DIR,
            )
            traditional_params = gr.TextArea(
                label='算法参数（例如 {"gamma": 0.8}）',
                value="",
            )
            tradition_btn = gr.Button("开始传统增强")
        with gr.Tab("深度增强"):
            checkpoint = gr.Textbox(label="模型 checkpoint 路径")
            device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="device")
            resize = gr.Number(label="resize（可选，边长，留空不缩放）")
            deep_save_dir = gr.Textbox(
                label="结果保存目录（留空默认 result/）", value=RESULT_DIR
            )
            deep_btn = gr.Button("开始深度增强")

    tradition_btn.click(
        run_tradition_inference,
        [input_img, algo, traditional_save_dir, traditional_params],
        [output_img, output_path],
    )
    deep_btn.click(
        run_deep_inference,
        [input_img, checkpoint, device, resize, deep_save_dir],
        [output_img, output_path],
    )

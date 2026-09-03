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


def _source(image, input_dir):
    if image is not None and (input_dir and input_dir.strip()):
        raise gr.Error("单张图像与文件夹输入不能同时使用，二选一")
    if image is not None:
        return image
    if input_dir and input_dir.strip():
        return input_dir.strip()
    raise gr.Error("请先上传图像或指定输入文件夹")


def _run(method, image, input_dir, save_dir, kwargs):
    source = _source(image, input_dir)
    out_dir = Path(save_dir.strip()) if save_dir and save_dir.strip() else DEFAULT_RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, str) and Path(source).is_dir():
        llv.predict(method, source, output=str(out_dir), **kwargs)
        return None

    out = out_dir / "output.png"
    _, saved = llv.predict(method, source, output=str(out), **kwargs)
    return llv.imread(saved, output_format="numpy")


def run_tradition_inference(image, input_dir, algo, save_dir, params):
    return _run(algo, image, input_dir, save_dir, _parse_params(params))


def run_deep_inference(image, input_dir, checkpoint, device, resize, save_dir):
    kwargs = {"device": device}
    if resize:
        kwargs["resize"] = int(resize)
    return _run(checkpoint, image, input_dir, save_dir, kwargs)


def build():
    with gr.Row(equal_height=True):
        with gr.Column():
            input_img = gr.Image(type="numpy", label="输入图像")
            input_dir = gr.Textbox(label="输入文件夹（与单张图像二选一，批量增强）")
        with gr.Column(scale=1):
            output_img = gr.Image(label="结果")

    with gr.Tabs():
        with gr.Tab("传统增强"):
            with gr.Row():
                algo = gr.Dropdown(ALGORITHMS, value=ALGORITHMS[0], label="算法")
                traditional_save_dir = gr.Textbox(
                    label="结果保存目录（留空默认 result/）",
                    value=RESULT_DIR,
                )
            traditional_params = gr.TextArea(
                label='算法参数（例如 {"gamma": 0.8}）',
                value="",
            )
            tradition_btn = gr.Button("开始增强")

        with gr.Tab("深度增强"):
            checkpoint = gr.Textbox(label="模型 checkpoint 路径")
            device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="device")
            resize = gr.Number(label="resize（可选，边长，留空不缩放）")
            deep_save_dir = gr.Textbox(
                label="结果保存目录（留空默认 result/）", value=RESULT_DIR
            )
            deep_btn = gr.Button("开始增强")

    tradition_btn.click(
        run_tradition_inference,
        [input_img, input_dir, algo, traditional_save_dir, traditional_params],
        [output_img],
    )
    deep_btn.click(
        run_deep_inference,
        [input_img, input_dir, checkpoint, device, resize, deep_save_dir],
        [output_img],
    )

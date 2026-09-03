import gradio as gr

from app import ALGORITHMS, RESULT_DIR
from app.inference.service import run_deep_inference, run_tradition_inference


def build():
    with gr.Row(equal_height=True):
        with gr.Column():
            input_img = gr.Image(type="numpy", label="输入图像", height=300)
            input_dir = gr.Textbox(label="输入文件夹（与单张图像二选一，批量增强）")
        with gr.Column(scale=1):
            output_img = gr.Image(label="结果", height=300)
            status = gr.Textbox(label="状态")

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
        [output_img, status],
    )
    deep_btn.click(
        run_deep_inference,
        [input_img, input_dir, checkpoint, device, resize, deep_save_dir],
        [output_img, status],
    )

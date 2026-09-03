import gradio as gr

from app import MODELS
from app.train.service import run_train


def build():
    """渲染训练页：左侧配置项、右侧超参，按钮触发训练。"""
    with gr.Row(equal_height=True):
        with gr.Column():
            model = gr.Dropdown(MODELS, value=MODELS[0], label="模型")
            root_dir = gr.Textbox(label="数据集根目录")
            train_save_dir = gr.Textbox(
                label="结果保存目录（留空默认 checkpoints/<Model>_<Dataset>/）"
            )
        with gr.Column():
            epochs = gr.Number(value=10, label="epochs")
            batch_size = gr.Number(value=4, label="batch_size")
            device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="device")

    train_btn = gr.Button("开始训练")
    train_out = gr.JSON(label="训练结果")
    train_btn.click(
        run_train,
        [model, root_dir, epochs, batch_size, device, train_save_dir],
        [train_out],
    )

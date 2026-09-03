import gradio as gr
import openLLV as llv

from app import MODELS


def run_train(model, root_dir, epochs, batch_size, device, save_dir):
    kwargs = {}
    if save_dir and save_dir.strip():
        kwargs["output_dir"] = save_dir.strip()
    return llv.train(
        model,
        root_dir=root_dir,
        epochs=int(epochs),
        batch_size=int(batch_size),
        device=device,
        **kwargs,
    )


def build():
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

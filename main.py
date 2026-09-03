import gradio as gr

from app.evaluate import build as build_evaluate
from app.inference import build as build_inference
from app.train import build as build_train

with gr.Blocks(title="openLLV WebUI") as demo:
    gr.Markdown("# openLLV WebUI")

    with gr.Tab("推理 Inference"):
        build_inference()

    with gr.Tab("训练 Train"):
        build_train()

    with gr.Tab("评估 Evaluate"):
        build_evaluate()


if __name__ == "__main__":
    demo.launch()

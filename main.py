import gradio as gr

from app import ALGORITHMS, DEFAULT_METRICS, METRICS, MODELS
from app.evaluate import run_evaluate
from app.inference import run_inference
from app.train import run_train

with gr.Blocks(title="openLLV WebUI") as demo:
    gr.Markdown("# openLLV WebUI")

    with gr.Tab("推理 Inference"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="numpy", label="输入图像")
                method = gr.Dropdown(ALGORITHMS, value=ALGORITHMS[0], label="传统算法")
                checkpoint = gr.Textbox(
                    label="深度模型 checkpoint 路径（可选，填了优先使用）"
                )
                infer_btn = gr.Button("开始推理")
            with gr.Column():
                output_img = gr.Image(label="结果")
                output_path = gr.Textbox(label="保存路径")
        infer_btn.click(
            run_inference, [input_img, method, checkpoint], [output_img, output_path]
        )

    with gr.Tab("训练 Train"):
        model = gr.Dropdown(MODELS, value=MODELS[0], label="模型")
        root_dir = gr.Textbox(label="数据集根目录")
        with gr.Row():
            epochs = gr.Number(value=10, label="epochs")
            batch_size = gr.Number(value=4, label="batch_size")
            device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="device")
        train_btn = gr.Button("开始训练")
        train_out = gr.JSON(label="训练结果")
        train_btn.click(
            run_train, [model, root_dir, epochs, batch_size, device], [train_out]
        )

    with gr.Tab("评估 Evaluate"):
        en_dir = gr.Textbox(label="增强图目录")
        ref_dir = gr.Textbox(label="参考图目录（可选，全参考指标需要）")
        metrics = gr.CheckboxGroup(METRICS, value=DEFAULT_METRICS, label="指标")
        eval_btn = gr.Button("开始评估")
        eval_out = gr.JSON(label="评估结果")
        eval_btn.click(run_evaluate, [en_dir, ref_dir, metrics], [eval_out])


if __name__ == "__main__":
    demo.launch()

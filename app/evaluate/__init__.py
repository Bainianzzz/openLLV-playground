import gradio as gr

from app import METRICS
from app.evaluate.service import run_evaluate


def build():
    """渲染评估页：增强图目录、参考图目录、指标、结果保存文件。"""
    en_dir = gr.Textbox(label="增强图目录")
    ref_dir = gr.Textbox(label="参考图目录（可选，全参考指标需要）")
    metrics = gr.CheckboxGroup(METRICS, value=METRICS[:2], label="指标")
    eval_save_dir = gr.Textbox(label="结果保存文件（留空默认 results/eval.json）")
    eval_btn = gr.Button("开始评估")
    eval_out = gr.JSON(label="评估结果")
    eval_btn.click(
        run_evaluate, [en_dir, ref_dir, metrics, eval_save_dir], [eval_out]
    )

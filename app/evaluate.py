import gradio as gr
import openLLV as llv

from app import METRICS


def run_evaluate(en_dir, ref_dir, metrics, save_dir):
    ref = ref_dir.strip() if ref_dir and ref_dir.strip() else None
    save_path = save_dir.strip() if save_dir and save_dir.strip() else None
    return llv.evaluate(
        en_img_dir=en_dir,
        ref_img_dir=ref,
        metrics=metrics,
        save_path=save_path,
    )


def build():
    en_dir = gr.Textbox(label="增强图目录")
    ref_dir = gr.Textbox(label="参考图目录（可选，全参考指标需要）")
    metrics = gr.CheckboxGroup(METRICS, value=METRICS[:2], label="指标")
    eval_save_dir = gr.Textbox(label="结果保存文件（留空默认 results/eval.json）")
    eval_btn = gr.Button("开始评估")
    eval_out = gr.JSON(label="评估结果")
    eval_btn.click(
        run_evaluate, [en_dir, ref_dir, metrics, eval_save_dir], [eval_out]
    )

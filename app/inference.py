import tempfile
from pathlib import Path

import gradio as gr
import openLLV as llv


def run_inference(image, method, checkpoint):
    if image is None:
        raise gr.Error("请先上传图像")
    target = checkpoint.strip() if checkpoint and checkpoint.strip() else method
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input.png"
        llv.imwrite(image, str(inp))
        _, saved = llv.predict(target, str(inp), output=str(Path(tmp) / "output.png"))
        result = llv.imread(saved, output_format="numpy")
        return result, str(saved)

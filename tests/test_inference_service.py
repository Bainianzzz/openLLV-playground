"""app.inference.validate 中自写逻辑的单元测试。"""

import gradio as gr
import pytest

from app.inference.validate import _parse_params, _source


class TestParseParams:
    """算法参数字符串的 JSON 解析逻辑。"""

    def test_valid(self):
        """空值返回空字典，合法 JSON 对象原样返回。"""
        assert _parse_params("") == {}
        assert _parse_params(None) == {}
        assert _parse_params('{"gamma": 0.8}') == {"gamma": 0.8}

    def test_invalid(self):
        """非法 JSON 或非对象（如数组）抛 gr.Error。"""
        with pytest.raises(gr.Error):
            _parse_params("not json")
        with pytest.raises(gr.Error):
            _parse_params("[1, 2, 3]")


class TestSource:
    """推理输入来源（单图 / 文件夹）的互斥与优先级逻辑。"""

    def test_valid(self):
        """仅单图返回图像，仅文件夹返回去空白路径。"""
        assert _source("img", "") == "img"
        assert _source(None, "  dir  ") == "dir"

    def test_invalid(self):
        """两者同时提供或都未提供时抛 gr.Error。"""
        with pytest.raises(gr.Error):
            _source("img", "dir")
        with pytest.raises(gr.Error):
            _source(None, "")

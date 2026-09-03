# openLLV WebUI - 实际实现归档

> 本文件记录当前代码的**实际实现情况**，是最终落地后的事实记录。随代码迭代更新。

## 1. 最终文件结构

```
openllv-playground/
├── main.py                 # Gradio 入口：gr.Blocks + 三个 Tab，只负责组装
├── app/
│   ├── __init__.py         # 从 cfg/*.yaml 加载配置并导出常量
│   ├── inference.py        # 推理页（传统增强 / 深度增强，单图或批量）
│   ├── train.py            # 训练页
│   └── evaluate.py         # 评估页
├── cfg/
│   ├── models.yaml         # algorithms / models / metrics 可选项
│   └── inference.yaml      # result_dir / checkpoint_dir 输出目录
├── docs/
│   └── archive/            # 文档归档（本文件）
├── Makefile                # sync / cpu / cuda / dev / start
├── pyproject.toml          # 依赖 + uv sources/index 配置
└── uv.lock
```

每个 `app/*.py` 暴露一个 `build()` 函数，在 `gr.Tab` 上下文内渲染该页组件并绑定事件；`main.py` 只顺序调用三个 `build()`。

## 2. 各功能实现

### 推理（`app/inference.py`）

单页内含两个子 Tab（`gr.Tabs`）：传统增强、深度增强，共用同一份输入/输出。

- **输入**：`gr.Image`（单张）+ `gr.Textbox` 输入文件夹，两者互斥（`_source` 校验，同给则 `gr.Error`）。
- **输出**：只显示一张 `gr.Image`；文件夹输入时返回 `None`（图不显示，文件落到输出目录）。
- **传统增强**：算法下拉（`ALGORITHMS`）+ 保存目录 + 算法参数（JSON 文本框），参数经 `json.loads` 解析后透传 `llv.predict`。
- **深度增强**：checkpoint 路径 + `device`（cpu/cuda）+ 可选 `resize`，透传 `llv.predict`。
- 单图输出文件默认 `output.png`；`input_dir` 为目录时按 `llv.predict` 行为递归输出到 `save_dir`。

### 训练（`app/train.py`）

左右等高两列：左列 模型 / 数据集根目录 / 结果保存目录；右列 epochs / batch_size / device。直接透传 `llv.train`，`save_dir` 非空时映射为 `output_dir`。

### 评估（`app/evaluate.py`）

输入 增强图目录 / 参考图目录（可选）/ 指标多选（默认前两个）/ 结果保存文件，透传 `llv.evaluate`。`save_dir` 非空时映射为 `save_path`。

## 3. 配置（`cfg/`）

`app/__init__.py` 用 `yaml.safe_load` 读取 `cfg/` 下各文件，导出：

| 常量 | 来源 | 值 |
| --- | --- | --- |
| `ALGORITHMS` | `models.yaml` | `he, clahe, gamma` |
| `MODELS` | `models.yaml` | `zerodce, sci` |
| `METRICS` | `models.yaml` | `PSNR, SSIM, MSE, MAE` |
| `RESULT_DIR` | `inference.yaml` | `results` |
| `CHECKPOINT_DIR` | `inference.yaml` | `checkpoints` |

## 4. Makefile 命令

```bash
make sync    # 自动检测：有 NVIDIA 驱动装 cu126，否则默认
make cpu     # uv sync（默认 PyPI）
make cuda    # 强制 cu126 源
make dev     # gradio 热更新启动（--watch-dirs app）
make start   # uv run python main.py
```

## 5. 实现说明 / 与早期设计的偏离

- 深度增强暴露了 `device`（早期规划为推理不暴露、由 openLLV 自动选择）。
- 可选项最初设计写在 `app/__init__.py` 数组里，实际改为 `cfg/*.yaml` 由 `__init__.py` 加载。
- 输出目录最初为 `result`，后与 `.gitignore`（`results/`、`checkpoints/`）统一为 `results`。
- 推理页支持了批量（文件夹输入），这是原始设计未明确的部分。

## 6. 运行

```bash
make start   # 或 uv run python main.py
```

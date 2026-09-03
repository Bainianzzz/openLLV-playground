# openLLV WebUI - MVP 实现过程

> 基于 openLLV 的 Gradio 图形化界面，覆盖推理、训练、评估三大功能。本文记录从零到 MVP 的完整实现路径与最终落地情况。

## 1. 概述

给 openLLV（低层视觉工具包）做图形化 Web 界面，让用户不写代码即可完成图像增强的推理、训练、评估。底层能力全部复用 openLLV，WebUI 只负责参数与结果可视化。

## 2. 实现步骤

1. **搭建环境**：uv 项目，`openllv`（git 依赖）+ `gradio` + `pyyaml`；torch 按平台区分（mac/win 走 PyPI，linux 走 pytorch index，可切 cu126）；`Makefile` 封装 `sync`/`cpu`/`cuda`/`dev`/`start`。

2. **设计**：先写产品设计与开发规格（`product-design.md`、`spec.md`），后续合并归档为本文，不再保留设计文档。

3. **实现三大功能**：`main.py` 只组装 `gr.Blocks` + 三个 `gr.Tab`；每个功能一个模块，业务方法 `run_*` 透传 openLLV API（`predict`/`train`/`evaluate`），界面 `build()` 渲染组件并绑定事件。

4. **结构重构**：每个功能由单文件改为包，界面组装 `build()` 与业务方法 `run_*` 分离，避免混在一起。

5. **配置下沉**：页面可选项与输出目录放 `cfg/*.yaml`，由 `app/__init__.py` 加载导出常量，页面不再硬编码。

6. **工程化**：补充 pytest 用例、pre-commit（ruff + pytest）、ruff D103 强制函数 docstring。

## 3. 最终文件结构

```
openllv-playground/
├── main.py                      # Gradio 入口：gr.Blocks + 三个 Tab，只组装
├── app/
│   ├── __init__.py              # 从 cfg/*.yaml 加载配置导出常量
│   ├── inference/
│   │   ├── __init__.py          # build() 界面组装
│   │   └── service.py           # run_* 业务方法
│   ├── train/
│   │   ├── __init__.py
│   │   └── service.py
│   └── evaluate/
│       ├── __init__.py
│       └── service.py
├── cfg/
│   ├── models.yaml              # algorithms / models / metrics
│   └── inference.yaml           # result_dir / checkpoint_dir
├── tests/
│   └── test_inference_service.py
├── docs/archive/                # 本文档
├── Makefile                     # sync / cpu / cuda / dev / start
├── .pre-commit-config.yaml      # ruff + pytest
├── AGENTS.md                    # 编码规范
├── pyproject.toml
└── uv.lock
```

## 4. 三大功能

### 推理

两个子 Tab（传统增强 / 深度增强）共用输入输出。

- 输入：`gr.Image`（单张）或 `gr.Textbox` 文件夹（批量），二者互斥。
- 输出：结果图 + 状态栏，完成后状态栏显示 `finish`；文件夹输入时结果图为空。
- 传统增强：算法下拉 + 算法参数（JSON 文本，解析后透传 `llv.predict`）。
- 深度增强：checkpoint 路径 + `device` + 可选 `resize`。

### 训练

左右等高两列（模型/数据集/保存目录 ｜ epochs/batch_size/device），透传 `llv.train`。

### 评估

增强图目录 / 参考图目录（可选）/ 指标多选 / 结果保存文件，透传 `llv.evaluate`。

## 5. 配置

| 常量 | 来源 | 值 |
| --- | --- | --- |
| `ALGORITHMS` | `models.yaml` | `he, clahe, gamma` |
| `MODELS` | `models.yaml` | `zerodce, sci` |
| `METRICS` | `models.yaml` | `PSNR, SSIM, MSE, MAE` |
| `RESULT_DIR` | `inference.yaml` | `results` |
| `CHECKPOINT_DIR` | `inference.yaml` | `checkpoints` |

## 6. 工程化

- **测试**：`tests/test_inference_service.py` 覆盖 `_parse_params`、`_source` 两个自写逻辑；透传 openLLV 的方法不测。
- **pre-commit**：提交时自动跑 `ruff check` + `pytest`，任一失败阻止提交。
- **lint**：ruff 启用 `D103`（公共函数缺 docstring），强制函数注释。

## 7. 运行

```bash
make start   # uv run python main.py
make dev     # gradio 热更新
```

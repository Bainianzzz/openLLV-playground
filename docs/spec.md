# openLLV WebUI 开发规格（Spec）

> 与 `product-design.md` 的区别：本文件只讲文件结构、功能与硬性要求，不框定具体代码实现。

## 1. 项目文件结构

```
openllv-playground/
├── main.py                 # 入口，组装 Gradio 界面并 launch
├── app/                    # 功能实现（三个模块，一个功能一个文件）
│   ├── __init__.py         # 可选项列表（算法/模型/指标）集中定义
│   ├── inference.py        # 推理
│   ├── train.py            # 训练
│   └── evaluate.py         # 评估
├── docs/                   # 文档
│   ├── product-design.md
│   └── spec.md
├── pyproject.toml          # 依赖：gradio、openllv
├── Makefile                # 环境切换：sync / cpu / cuda
└── uv.lock
```

结构不是强制约束，唯一要求是**三大功能各占一个模块**，`main.py` 只负责界面组装，不写业务逻辑。

## 2. 功能

三个 Tab，对应 openLLV 三个顶层 API：

| 功能 | 模块 | 输入 | 输出 |
| --- | --- | --- | --- |
| 推理 | `app/inference.py` | 图像 + 传统算法名或 checkpoint 路径 | 增强图 + 保存路径 |
| 训练 | `app/train.py` | 模型 + 数据集目录 + epochs/batch_size/device | 训练结果（checkpoint 路径、历史） |
| 评估 | `app/evaluate.py` | 增强图目录 + 参考图目录(可选) + 指标 | 指标得分与统计 |

## 3. 核心要求

1. **只复用 openLLV，不自造逻辑。** 图像读写、预测、训练、评估全部走 `openLLV` 的公开 API（`predict` / `train` / `evaluate` / `imread` / `imwrite` / `list_*`）。

2. **不加多余验证。** 不要自己写参数校验、路径检查、格式判断。非法输入直接交给 openLLV，让它自己 `raise`，把异常信息原样透传给用户（Gradio 端展示报错信息即可）。唯一的界面级判断是“用户是否填了必填项”，其余一律不拦。

3. **传统算法开箱即用，深度模型需 checkpoint。** 推理页默认列出 `list_algorithms()`；深度模型通过填写 checkpoint 路径触发，不提供“用随机权重跑模型”的入口。

4. **单机单用户。** 无登录、无任务队列、无并发隔离；训练等耗时操作直接阻塞运行，不做异步。

5. **device 交给用户选。** 训练暴露 device（cpu/cuda）；推理不暴露，交给 openLLV 自动选择。

6. **可选项集中定义，先做最小子集。** 界面下拉框的选项统一以数组形式放在 `app/__init__.py`，MVP 不列全量，只挑能验证功能的最小集合：

   - 传统算法：`he`、`clahe`、`gamma`
   - 深度模型：`zerodce`、`sci`
   - 评估指标：`PSNR`、`SSIM`、`MSE`、`MAE`（全参考，不依赖 pyiqa）

   后续扩充只需往数组加项。

## 4. 明确不做（MVP 之外）

- 不实现用户系统 / 鉴权。
- 不暴露细粒度训练配置（loss、optimizer、scheduler 等）。
- 不做批量任务队列、历史记录、结果持久化数据库。
- 不对输出图做额外的后处理或美化。

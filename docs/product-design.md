# openLLV WebUI 产品设计

## 1. 产品定位

给 openLLV（低层视觉工具包）做一个图形化 Web 界面，让用户不写代码就能完成图像增强的**推理、训练、评估**。底层能力完全复用 openLLV，WebUI 只负责把参数和结果可视化。

## 2. 三大功能（MVP）

| 功能 | openLLV API | 用户做什么 | 得到什么 |
| --- | --- | --- | --- |
| 推理 | `llv.predict` | 上传图像 + 选传统算法或填模型 checkpoint | 增强后的图像 |
| 训练 | `llv.train` | 选模型 + 填数据集路径 + 设超参 | checkpoint 与训练历史 |
| 评估 | `llv.evaluate` | 填增强图目录（可选参考图目录）+ 选指标 | 每张图的得分与统计均值 |

## 3. 关键 API 摘要

```python
import openLLV as llv

# 推理：返回 (增强图, 保存路径)
llv.predict(method, source, output=None, **kwargs)

# 训练：返回 {"checkpoint_dir", "history", "best_val_loss", ...}
llv.train(model, root_dir=..., epochs=..., batch_size=..., device=...)

# 评估：返回 {"filenames", "metrics", "statistics", ...}
llv.evaluate(en_img_dir, ref_img_dir=None, metrics=[...])

# 可选项列表
llv.list_algorithms()   # 传统算法
llv.list_models()       # 深度学习模型
llv.list_metrics()      # 评估指标
```

两条后端路线：

- **传统算法**（`list_algorithms`：HE、CLAHE、Gamma、MSR、NPE、LIME 等）——开箱即用，无需权重。
- **深度学习模型**（`list_models`：ZeroDCE、SCI、KinD、RetinexFormer 等）——需要 openLLV checkpoint 才有意义，否则是随机权重。

## 4. 界面设计

三个 Tab，对应三大功能。

**推理**：输入图（上传）+ 算法下拉框（传统算法）+ checkpoint 路径文本框（填了则优先走深度模型）→ 输出图 + 保存路径。

**训练**：模型下拉框 + 数据集根目录 + epochs / batch_size / device → 训练结果 JSON。数据集目录结构：

```
dataset_root/
  train/input/   train/target/
  val/input/     val/target/
```

文件名按大小写不敏感的 stem 配对。

**评估**：增强图目录 + 参考图目录（可选）+ 指标多选 → 结果 JSON。指标分两类：

- 全参考（需要参考图）：PSNR、SSIM、MSE、MAE、LPIPS、LOE
- 无参考：NIQE、MUSIQ、PI

LPIPS / NIQE / MUSIQ / PI 依赖 pyiqa，首次运行可能下载权重。

## 5. 技术栈

- 后端：openLLV（`openLLV` git 依赖，自带 torch）
- 界面：Gradio
- 包管理：uv（CPU / CUDA 通过 `make sync` / `make cuda` 切换）

## 6. MVP 边界

- 推理默认只提供传统算法；深度模型必须用户自备 checkpoint。
- 训练只暴露常用超参（模型、数据集、epochs、batch_size、device），不做细粒度 loss/optimizer 配置。
- 评估默认全参考指标 PSNR/SSIM；pyiqa 类指标按需勾选。
- 无用户系统、无任务队列，单机单用户。

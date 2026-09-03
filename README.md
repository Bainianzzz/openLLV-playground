# openLLV-playground

openLLV 的 Gradio WebUI，覆盖推理、训练、评估三大功能。

## 快速开始

```bash
make sync    # 安装依赖（自动检测：有 NVIDIA 驱动装 cu126，否则默认）
make start   # 启动服务
```

其他命令：

```bash
make cpu     # 强制 CPU 版 PyTorch
make cuda    # 强制 CUDA 12.6 版 PyTorch
make dev     # 开发模式（gradio 热更新）
```

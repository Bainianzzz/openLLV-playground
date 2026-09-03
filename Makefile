CUDA_INDEX = https://download.pytorch.org/whl/cu126

.PHONY: sync cpu cuda dev

# 自动检测：有 NVIDIA 驱动就装 cu126，否则默认 PyPI
sync:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "CUDA detected -> $(CUDA_INDEX)"; \
		UV_INDEX="pytorch=$(CUDA_INDEX)" uv sync; \
	else \
		echo "No CUDA -> default PyPI"; \
		uv sync; \
	fi

cpu:
	uv sync

cuda:
	UV_INDEX="pytorch=$(CUDA_INDEX)" uv sync

dev:
	uv sync --group dev

- [1. 常见问题](#1-常见问题)
  - [1.1 找不到'py_utils'](#11-找不到py_utils)
  - [1.2 外部grq量化失败](#12-外部grq量化失败)
  - [1.3 导出Tokenizer失败](#13-导出Tokenizer失败)

## 1. 常见问题

### 1.1 找不到'py_utils'

当出现`ModuleNotFoundError: No module named 'py_utils'`说明此时python环境变量设置错误，需要正确设置环境变量。请根据实际rknn3-model-zoo目录路径进行设置。

例如，如果rknn3-model-zoo位于`/home/rockchip/rknn3-model-zoo`，则设置为：
```bash
export PYTHONPATH="/home/rockchip/rknn3-model-zoo:$PYTHONPATH"
echo $PYTHONPATH # 确认环境变量设置正确
```

### 1.2 外部grq量化失败

首先检查环境中是否存在CUDA，外部grq量化仅支持在CUDA环境中运行；其次外部grq量化仅支持主流模型，如Qwen、LLama、MiniCPM等，遇到不支持的模型可关闭外部grq量化使用rknn3-toolkit工具进行量化。

### 1.3 导出Tokenizer失败

工程默认使用`convert_hf_to_gguf.py`导出model.tokenizer.gguf文件，对于新模型可能存在导出失败的情况，需要对应修改`convert_hf_to_gguf.py`文件。


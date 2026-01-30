# 1. FAQ (Frequently Asked Questions)

### 1.1 Cannot find 'py_utils'

If you encounter a `ModuleNotFoundError: No module named 'py_utils'`, it indicates that the Python environment variable is set incorrectly. You need to set the environment variable properly based on the actual path of your `rknn3-model-zoo` directory.

For example, if `rknn3-model-zoo` is located at `/home/rockchip/rknn3-model-zoo`, set the environment variable as follows:

```Bash
export PYTHONPATH="/home/rockchip/rknn3-model-zoo:$PYTHONPATH"
echo $PYTHONPATH # Confirm the environment variable is set correctly
```

### 1.2 External GRQ Quantization Fails

First, check if CUDA is available in your environment, as external GRQ quantization is only supported in a CUDA environment.

Second, external GRQ quantization only supports mainstream models, such as Qwen, Llama, MiniCPM, etc. If you encounter an unsupported model, you can disable external GRQ quantization and use the `rknn3-toolkit` for quantization instead.

### 1.3 Tokenizer Export Fails

The project uses `convert_hf_to_gguf.py` by default to export the `model.tokenizer.gguf` file. This process may fail for new models. If it does, you will need to modify the `convert_hf_to_gguf.py` file accordingly.

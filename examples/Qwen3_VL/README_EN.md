# Qwen3-VL Model Deployment Guide

## 1. Environment Requirements

Qwen3-VL has specific dependency requirements that are incompatible with the `requirements.txt` in this repository. Please install the following dependencies manually:

```
torch >= 2.9.0
transformers == 4.57.1
onnxruntime >= 1.23.2
```

> ⚠️ Using the default dependencies will cause model conversion to fail. Please install the versions specified above.

## 2. Model Pruning Strategy

To support larger context lengths, appropriate pruning is required when deploying large-scale multimodal models.

### 2.1 Vision Model Pruning

Some operators are offloaded to the CPU of the host device (e.g., RK3588).

### 2.2 LLM Model Pruning

The LLM Head is separated and runs independently on the host device, reducing memory usage on the coprocessor. (Optional)

### 2.3 Full Model Mode (No Pruning)

Devices with larger memory (e.g., RK1828) can use the full model directly. Add the `--no_prune_mode` parameter when exporting:

```bash
python export_rknn.py --no_prune_mode
```

## 3. Supported Models

Currently supports Qwen3-VL 2B, 4B, and other variants. Specify the corresponding model path when exporting.

Example with **Qwen3-VL-4B**:

```bash
# Export ONNX model
python export_llm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_llm_path Qwen3-VL-4B-llm.onnx \
    --modelscope

# Export RKNN model
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn
```

## 4. Vision Model Resolution Adjustment

Use `--img_h` and `--img_w` parameters to adjust input resolution (must be a multiple of 32):

```bash
# Export Vision ONNX model
python export_vision.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384 \
    --modelscope

# Export Vision RKNN model
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn
```

> ⚠️ **Note**:
> - Higher resolution increases memory usage and affects the maximum LLM context length
> - Some resolutions may be incompatible with the RKNN inference framework; contact the RKNPU team if errors occur

## 5. C++ Deployment

The C++ inference code automatically detects the model format and is compatible with both pruned and full models without code modification.

The C++ demo remains compatible with the original command line. To enable SpeedUP,
append the optional `speedup_ratio` argument:

```bash
./rknn_qwen3_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` values:
- `1.0`: automatic mode.
- `0.0`: disabled.
- `(0.0, 1.0)`: manual mode.

If you modified the Vision model resolution, update the parameters in `rknn_qwen3_vl_vision.h`:

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

## 6. Troubleshooting

### ONNX File Path Issue

Models exported with PyTorch ≥ 2.9.0 generate both `xxx.onnx` and `xxx.onnx.data` files. When running `rknn.load_llm`, ensure both files are in the same directory, otherwise you'll get:

```
RUNTIME_EXCEPTION: Exception during initialization: filesystem error: 
cannot get file size: No such file or directory [Qwen3-VL-4B-llm.onnx.data]
```

### PyTorch Version Incompatibility

If using PyTorch < 2.9.0, `rknn.load_llm` will fail with:

```
RUNTIME EXCEPTION : Non-zero status code, returned while running Reshape node. 
...
The input tensor cannot be reshaped to the requested shape. Input shape:{384}, requested shape:{64,1}
```

Please upgrade PyTorch to ≥ 2.9.0.

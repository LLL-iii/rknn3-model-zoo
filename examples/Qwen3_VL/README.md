# Qwen3-VL 模型部署说明

## 1. 部署环境

由于 Qwen3-VL 模型对依赖库版本有特殊要求，与本仓库 `requirements.txt` 中的版本不兼容，请手动安装以下依赖：

```
torch >= 2.9.0
transformers == 4.57.1
onnxruntime >= 1.23.2
```

> ⚠️ 使用默认依赖会导致模型转换失败，请务必按上述版本安装。

## 2. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时需进行适当裁剪。

### 2.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 2.2 LLM 模型裁剪

将 LLM Head 独立出来，在主控设备上单独运行，从而减少协处理器的内存占用。（可选）

### 2.3 完整模型模式（无裁剪）

RK1828 等内存较大的设备可直接使用完整模型，导出时添加 `--no_prune_mode` 参数：

```bash
python export_rknn.py --no_prune_mode
```

## 3. 支持的模型

目前支持 Qwen3-VL 2B 和 4B 等模型。导出时请指定对应的模型路径。

以 **Qwen3-VL-4B** 为例：

```bash
# 导出 ONNX 模型
python export_llm.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_llm_path Qwen3-VL-4B-llm.onnx \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn
```

## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 32 的倍数）：

```bash
# 导出 Vision ONNX 模型
python export_vision.py \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384 \
    --modelscope

# 导出 Vision RKNN 模型
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn
```

> ⚠️ **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

## 5. C++ 部署说明

C++ 推理代码已实现模型格式自动识别，无需修改代码即可兼容裁剪版与完整版模型。

C++ demo 保持原有命令行参数兼容；如需启用 SpeedUP，可在命令末尾增加可选参数
`speedup_ratio`：

```bash
./rknn_qwen3_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` 取值说明：
- `1.0`：自动模式。
- `0.0`：关闭 SpeedUP。
- `(0.0, 1.0)`：手动模式。

若修改了 Vision 模型的分辨率，需同步调整 `rknn_qwen3_vl_vision.h` 中的参数：

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

## 6. 常见问题

### ONNX 文件路径问题

使用 PyTorch ≥ 2.9.0 导出的模型会生成 `xxx.onnx` 和 `xxx.onnx.data` 两个文件。执行 `rknn.load_llm` 时，必须确保这两个文件在同一目录下，否则会报错：

```
RUNTIME_EXCEPTION: Exception during initialization: filesystem error: 
cannot get file size: No such file or directory [Qwen3-VL-4B-llm.onnx.data]
```

### PyTorch 版本不兼容

若使用 PyTorch < 2.9.0，执行 `rknn.load_llm` 时会报错：

```
RUNTIME EXCEPTION : Non-zero status code, returned while running Reshape node. 
...
The input tensor cannot be reshaped to the requested shape. Input shape:{384}, requested shape:{64,1}
```

请升级 PyTorch 至 ≥ 2.9.0 版本。

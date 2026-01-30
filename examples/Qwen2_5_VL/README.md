# Qwen2.5-VL 模型部署说明

## 1. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时可进行适当裁剪。

### 1.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 1.2 LLM 模型裁剪

将 LLM Head 独立出来，在主控设备上单独运行，从而减少协处理器的内存占用。（可选）

### 1.3 完整模型模式（无裁剪）

RK1828 等内存较大的设备可直接使用完整模型，导出时添加 `--no_prune_mode` 参数：

```bash
python export_rknn.py --no_prune_mode
```

## 2. 支持的模型

目前支持 Qwen2.5-VL 3B 和 7B 等模型。导出时请指定对应的模型路径。

以 **Qwen2.5-VL-7B** 为例：

```bash
# 导出 LLM ONNX 模型
python export_llm.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --export_llm_path ../../model/llm/Qwen2.5-VL-7B-llm.onnx \
    --modelscope

# 导出 LLM RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/Qwen2.5-VL-7B-llm.onnx \
    --config ../../model/llm/Qwen2.5-VL-7B-llm.config.pkl \
    --rknn_path ../../model/llm/Qwen2.5-VL-7B-llm.rknn

# 导出 Vision ONNX 模型
python export_vision.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --export_vision_path ../../model/vision/Qwen2.5-VL-7B-vision.onnx \
    --modelscope

# 导出 Vision RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/vision/Qwen2.5-VL-7B-vision.onnx \
    --rknn_path ../../model/vision/Qwen2.5-VL-7B-vision.rknn
```

## 3. KV Cache INT4 量化
在大规模语言模型推理过程中，KV Cache（Key/Value Cache）用于存储历史的注意力键值，以避免重
复计算，从而提高推理速度。随着序列长度增长，KV Cache 的内存占用会快速增加。为了减少 KV
Cache 的存储带宽与内存访问开销，可以采用量化方式将其从 FP16/FP32 转换为 INT8 或更低位宽表
示。但由于 KV Cache 数值分布随时间逐 token 动态变化，如果对整段 KV 使用统一的量化参数，会导致
量化误差累积从而影响推理精度。因此通常采用分组量化（Group Quantization）来降低精度损失。
目前，RKNN 的LLM支持两种 KV Cache 量化模式：
Int8_to_F16（默认）：以 INT8 格式存储，计算时转换回 FP16；
Int4_to_F16（适用于更长上下文场景,有一定精度损失）：以 INT4 格式存储，计算时转换回 FP16。
若需支持更长的上下文长度并进一步压缩 KV Cache 内存，建议启用 Int4_to_F16 模式。
启用 Int4_to_F16 的 RKNN 模型转换配置如下：
```python
rknn.config(target_platform='rk1820', 
          quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
          max_ctx_len           =2048,
          max_position_embeddings=2048,
          kvcache_store_method='GroupQuant', kvcache_dtype='Int4_to_F16', 
          kvcache_group_size=16, kvcache_residual_depth=64,
          )
```
- 注意：上述配置位于 python/llm/export_rknn.py 文件中，请根据实际需求调整相关参数。


## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 28 的倍数）：

```bash
python export_vision.py --img_h 392 --img_w 392
```

> ⚠️ **注意**：
> - 裁剪版本不支持修改分辨率
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

## 5. C++ 部署说明

C++ 推理代码已实现模型格式自动识别，无需修改代码即可兼容裁剪版与完整版模型。

若修改了 Vision 模型的分辨率，需同步调整 `rknn_qwen2_5_vl_vision.h` 中的参数：

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

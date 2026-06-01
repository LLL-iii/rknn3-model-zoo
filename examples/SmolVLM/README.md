# SmolVLM 模型部署说明

模型地址：[HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)

## 1. 导出 RKNN 模型

### 1.1 Vision 模型

```bash
cd python/vision

# 导出 ONNX 模型
python export_vision.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

### 1.2 LLM 模型

```bash
cd python/llm

# 导出 ONNX 模型
python export_llm.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

> 导出完成后，模型文件将生成在 `model` 目录下。

## 2. C++ 板端部署

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d SmolVLM
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_SmolVLM_demo/` 目录：

```
rknn_SmolVLM_demo/
├── lib/
│   ├── librga.so
│   └── librknn3_api.so
├── model/
│   ├── demo.jpg
│   ├── SmolVLM-500M-llm.rknn
│   ├── SmolVLM-500M-llm.weight
│   ├── SmolVLM-500M-llm.embed.bin
│   ├── SmolVLM-500M-llm.tokenizer.gguf
│   ├── SmolVLM-500M-vision.rknn
│   └── SmolVLM-500M-vision.weight
└── rknn_smol_vl_demo
```

### 2.2 部署到开发板

```bash
# 推送 demo 目录
adb push rknn_SmolVLM_demo /data/

# 推送运行库
adb push rknn_SmolVLM_demo/lib/* /usr/lib/
```

### 2.3 运行示例

```bash
adb shell
cd /data/rknn_SmolVLM_demo
```
运行命令如下：

```bash
./rknn_smol_vl_demo model/SmolVLM-500M-vision.rknn model/SmolVLM-500M-vision.weight model/SmolVLM-500M-llm.rknn model/SmolVLM-500M-llm.weight model/SmolVLM-500M-llm.tokenizer.gguf model/SmolVLM-500M-llm.embed.bin 0xff 0xff demo.jpg "Briefly describe this image?"
```

输出示例：
```
 A spaceman is drinking a beer on the moon.
```

## 3. GRQ量化
GRQ 量化是 RKNN 为 w4a16 精度的 LLM 模型专门设计的一种量化方法，其思路类似于 GPTQ 量化。经过实际部署测试后发现采用默认的 MMBench 数据集进行 GRQ 量化时，模型的精度会下降。因此，建议根据实际应用场景自行构建校准数据集。
你可以参考以下路径中的示例数据集作为构建参考：`rknn3_model_zoo/datasets/MMBench/llm/dataset.json`
该数据集需为 JSON 格式，结构如下：
```sh
[
    {"image_path": "../vision/images", "image": "1555.jpg", "input": "Briefly describe this image"},
    {"image_path": "../vision/images", "image": "1.jpg", "input": "Briefly describe this image"}
]
```
注意：
1. 每个样本必须包含以下三个字段：
    - image_path：图像所在目录的路径；
    - image：具体的图像文件名；
    - input：对应的文字输入提示（prompt）。
2. 由于RKNN3 Toolkit内部实现的GRQ量化算法限制，在使用量化时每个样本中图像 token 数与文本 token 数之和不得超过 2048。建议将图像缩放至视觉模型实际所需的输入尺寸，避免直接使用原始高分辨率图像，对文字输入prompt进行精简，控制其长度。该限制仅适用于RKNN3 Toolkit内部GRQ量化阶段，外部Torch版本的GRQ量化和板端推理无此限制。

在 RKNN 配置中开启 GRQ 量化，可使用如下代码：
```py
rknn.config(target_platform='rk1820', 
            max_ctx_len           =4096,
            max_position_embeddings=4096,
            quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32')
```
请确保校准数据集贴合实际推理场景，以获得最佳量化效果。
[English](README.md)

# RKNN3-MODEL-ZOO

RKNN3 SDK 提供了将 AI 模型部署到 RK1820/RK1828 协处理器所需的完整软件栈，包括：

- **[RKNN3-Toolkit](https://github.com/airockchip/rknn3-toolkit)**：PC 端软件开发套件，支持模型转换、推理和性能评估等。
- **RKNN3 Runtime**：板端运行时库，提供 C/C++ 编程接口，用于部署 RKNN 模型并加速 AI 应用。
- **[RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)**：模型转换与部署示例仓库，包含 CNN / LLM / VLM 等多种模型的参考实现。

本仓库提供了完整的模型部署流程：
- **模型导出**：将 HuggingFace / PyTorch 模型导出为 ONNX 格式
- **模型转换**：使用 RKNN3 Toolkit 将 ONNX 等模型转换为 RKNN 格式
- **板端部署**：提供 C++ 推理示例代码

## 1. 支持的模型

### LLM/VLM

- FastVLM
- SmolVLM / SmolVLM2
- GLM-Edge
- GME-Qwen2-VL
- InternVLM
- Janus-Pro
- MiniCPM-V / MiniCPM-V-4
- Qwen2.5 / Qwen2.5-VL / Qwen2.5-Omni
- Qwen3 / Qwen3-VL / Qwen3-VL-LoRA
- Gemma-4


### ASR（语音识别）

- Qwen3-ASR（流式/非流式）
- SenseVoiceSmall
- Zipformer

### TTS（语音合成）

- Qwen3-TTS
- VITS（LJSpeech / VCTK）

### Embedding / Reranker（语义向量与排序）

- Qwen3-Embedding
- Qwen3-Reranker

### 翻译

- HY-MT1.5

## OCR

- PaddleOCR-VL

### CV（计算机视觉）

- MobileNetV1 / MobileNetV2 / ResNet
- YOLOv5 / YOLOv6 / YOLOv8
- DINOv3
- SigLIP2

## 2. 支持的平台

| 主芯片 | 协处理器 | 操作系统 |
|--------|----------|----------|
| RK3588 系列 | RK1820 / RK1828 | Linux / Android |
| RK3576 系列 | RK1820 / RK1828 | Linux / Android |
| RK3572 系列 | - | Linux / Android |

> **构建与运行时库说明**：
> - 顶层构建脚本 `build-linux.sh` / `build-android.sh` 的 `-t` 参数支持 `rk3588`、`rk3576`、`rk3572`、`x86`。
> - 示例安装目录 `lib/` 中的 RKNN3 Runtime 库按 SoC 区分：
>   - `RK3588` / `RK3576`：安装 `librknn3_api.so` 和 `librknn3_api_rkcp.so`
>   - `RK3572`：安装 `librknn3_api.so` 和 `librknn3_api_native.so`

## 3. LLM 模型部署（以 Qwen2.5-3B 为例）

### 3.1 导出模型配置

> **环境要求**：Python 3.10

```bash
cd rknn3_model_zoo/
pip install -r requirements.txt
export PYTHONPATH=./

cd examples/Qwen2_5/python/
python export_llm.py --quant
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--load_weight` | bool | 加载模型权重（`False` 时仅导出模型结构） | `True` |
| `--quan_dataset` | bool | 生成量化数据集（需 `--load_weight=True`） | `True` |
| `--model_path` | str | 模型路径或 HuggingFace 名称 | `Qwen/Qwen2.5-3B-Instruct` |
| `--export_llm_path` | str | 导出路径 | `../model/llm/Qwen2.5-3B-Instruct.onnx` |
| `--quant` | bool | 启用 GRQ 量化算法 | `False` |
| `--modelscope` | bool | 从 ModelScope 下载模型（国内用户推荐） | `False` |

> **说明**：
> - 使用 GRQ 量化时，模型自带量化参数，转换 RKNN 时无需量化数据集
> - 导出的配置文件包括：ONNX 模型、Config、Tokenizer、Embed 文件

### 3.2 导出 RKNN 模型

```bash
python export_rknn.py
```

#### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--onnx_path` | str | 输入 ONNX 模型路径 | `../model/llm/Qwen2.5-3B-Instruct.onnx` |
| `--config` | str | 模型 Config 路径 | `../model/llm/Qwen2.5-3B-Instruct.config.pkl` |
| `--rknn_path` | str | 输出 RKNN 模型路径 | `../model/llm/Qwen2.5-3B-Instruct.rknn` |
| `--dataset_path` | str | 量化数据集路径 | `../data/dataset.txt` |

> **说明**：
> - 使用 GRQ 量化时，转换 RKNN 会忽略量化数据集
> - 导出采用权重分离模式，生成 `.rknn` 和 `.weight` 两个文件

### 3.3 Linux 平台示例

#### 编译

```bash
cd ../../../

# 可选：指定交叉编译器路径
export GCC_COMPILER=<GCC_COMPILER_PATH>

# 编译命令
./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d Qwen2_5

# 示例：RK3588 平台
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5
```

#### 推送到开发板

```bash
adb push install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_Qwen2_5_demo/ /data/
```

#### 运行

```bash
adb shell
cd /data/rknn_Qwen2_5_demo

export LD_LIBRARY_PATH=./lib
./rknn_qwen2_5_demo \
    model/Qwen2.5-0.5B-Instruct.rknn \
    model/Qwen2.5-0.5B-Instruct.weight \
    model/Qwen2.5-0.5B-Instruct.tokenizer.gguf \
    model/Qwen2.5-0.5B-Instruct.embed.bin \
    0xff \
    "Who are you?"
```

## 4. VLM 模型部署（以 FastVLM-1.5B 为例）

### 4.1 导出模型配置

> **环境要求**：Python 3.10

```bash
cd rknn3_model_zoo/
pip install -r requirements.txt
export PYTHONPATH=./

cd examples/FastVLM/python/

# 导出 LLM 模型配置
cd ./llm
python export_llm.py --quant

# 导出 Vision 模型配置
cd ../vision/
python export_vision.py
```

#### LLM 导出参数（export_llm.py）

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--load_weight` | bool | 加载模型权重（`False` 时仅导出模型结构） | `True` |
| `--quan_dataset` | bool | 生成量化数据集（需 `--load_weight=True`） | `True` |
| `--model_path` | str | 模型路径或 HuggingFace 名称 | `../../llava-fastvithd_1.5b_stage3` |
| `--export_llm_path` | str | 导出路径 | `../../model/llm/FastVLM-llm.onnx` |
| `--quant` | bool | 启用 AWQ + GRQ 量化算法 | `False` |
| `--modelscope` | bool | 从 ModelScope 下载模型（国内用户推荐） | `False` |

> **说明**：
> - 模型需从外部下载
> - 使用 GRQ 量化时，模型自带量化参数，转换 RKNN 时无需量化数据集
> - 导出的配置文件包括：ONNX 模型、Config、Tokenizer、Embed 文件

#### Vision 导出参数（export_vision.py）

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--load_weight` | bool | 加载模型权重 | `True` |
| `--model_path` | str | 模型路径 | `../../llava-fastvithd_1.5b_stage3` |
| `--export_vision_path` | str | 导出路径 | `../../model/vision/FastVLM-vision.onnx` |
| `--img_size` | int | 输入图像分辨率 | `512` |

> **说明**：Vision 模型仅导出 ONNX 格式

### 4.2 导出 RKNN 模型

```bash
# 导出 Vision RKNN 模型
cd ./vision
python export_rknn.py

# 导出 LLM RKNN 模型
cd ../llm
python export_rknn.py
```

#### Vision RKNN 导出参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--onnx_path` | str | 输入 ONNX 模型路径 | `../../model/vision/FastVLM-vision.onnx` |
| `--rknn_path` | str | 输出 RKNN 模型路径 | `../../model/vision/FastVLM-vision.rknn` |
| `--dataset_path` | str | 量化数据集路径 | `../../../../datasets/MMBench/vision/datasets.txt` |

> **说明**：默认使用 Normal 量化算法，可设置 `quantized_algorithm='grq'` 以提高精度

#### LLM RKNN 导出参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--onnx_path` | str | 输入 ONNX 模型路径 | `../../model/llm/FastVLM-llm.onnx` |
| `--config` | str | 模型 Config 路径 | `../../model/llm/FastVLM-llm.config.pkl` |
| `--rknn_path` | str | 输出 RKNN 模型路径 | `../../model/llm/FastVLM-llm.rknn` |
| `--dataset_path` | str | 量化数据集路径 | `../../data/llm/dataset.txt` |

> **说明**：
> - 使用 GRQ 量化时，转换 RKNN 会忽略量化数据集
> - 导出采用权重分离模式，生成 `.rknn` 和 `.weight` 两个文件

### 4.3 Linux 平台示例

#### 编译

```bash
cd rknn3_model_zoo/

# 可选：指定交叉编译器路径
export GCC_COMPILER=<GCC_COMPILER_PATH>

# 编译命令
./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d FastVLM

# 示例：RK3588 平台
./build-linux.sh -t rk3588 -a aarch64 -d FastVLM
```

#### 推送到开发板

```bash
adb push install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_FastVLM_demo/ /data/
```

#### 运行

```bash
adb shell
cd /data/rknn_FastVLM_demo

export LD_LIBRARY_PATH=./lib
./rknn_fastvlm_demo \
    model/FastVLM-vision.rknn \
    model/FastVLM-vision.weight \
    model/FastVLM-llm.rknn \
    model/FastVLM-llm.weight \
    model/FastVLM-llm.tokenizer.gguf \
    model/FastVLM-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "Please describe the content of the picture."
```
### 4.4 SpeedUP 用法

本仓库的 Qwen2.5-VL 和 Qwen3-VL 示例可链接 SpeedUP 第三方库，用于推理加速。

#### 文件位置

发布包中需要保留以下文件：

```text
3rdparty/SpeedUP/
├── include/speedup.h
├── Linux/aarch64/libSpeedUP.so
└── Android/arm64-v8a/libSpeedUP.so
```

#### 编译

Qwen2.5-VL：

```bash
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5_VL
```

Qwen3-VL：

```bash
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL
```

编译完成后，`libSpeedUP.so` 会安装到对应 demo 的 `lib/` 目录。

#### 运行参数

Qwen2.5-VL：

```bash
./rknn_qwen2_5_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

Qwen3-VL：

```bash
./rknn_qwen3_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` 为可选参数：

| 参数值 | 模式 |
|--------|------|
| `1.0` | 自动模式 |
| `0.0` | 关闭 |
| `(0.0, 1.0)` | 手动模式 |

RKNN3 多核设备通常可使用：

```bash
0xff 0xff
```

#### 运行示例

Qwen2.5-VL：

```bash
cd /userdata/rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen2_5_VL_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./rknn_qwen2_5_vl_demo \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-vision.rknn \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-vision.weight \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.rknn \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.weight \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.tokenizer.gguf \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.embed.bin \
    0xff 0xff \
    /userdata/rknn3-model-zoo/examples/Qwen2_5_VL/data/vision/demo.jpg \
    "请描述这张图片" \
    392 392 \
    1.0
```

Qwen3-VL：

```bash
cd /userdata/rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_VL_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./rknn_qwen3_vl_demo \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-vision_384_384.rknn \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-vision_384_384.weight \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.rknn \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.weight \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.tokenizer.gguf \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.embed.bin \
    0xff 0xff \
    /userdata/rknn3-model-zoo/examples/Qwen2_5_VL/data/vision/demo.jpg \
    "请描述这张图片" \
    384 384 \
    1.0
```

## 5. 模型适配指南

- **同系列模型兼容**：同一系列的示例程序相互兼容。例如，Qwen2.5-0.5B 的示例可直接用于 Qwen2.5-7B，只需修改模型加载路径。

- **新模型适配**：对于本仓库未收录的 LLM 模型，请参考 [LLM 模型适配教程](LLM_model_modification_guide_CN.md) 进行 ONNX 导出和部署移植。

## 6. 注意事项

- **Transformers 版本**：不同模型依赖的 `transformers` 版本可能不同，导出 ONNX 前请安装对应版本。版本信息可从模型的 config.json（例如 https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json ）中的 `transformers_version` 字段获取。部分模型有特殊版本要求，见各示例目录下的 `requirements.txt`。

- **PyTorch 版本**：建议使用 PyTorch <= 2.8.0（Qwen3-VL、Gemma-4 等模型需 PyTorch >= 2.9.0，PaddleOCR-VL 需 transformers == 4.55.0，具体看对应模型下的 requirements.txt）

- **模块兼容性**：Gemma-4 的 Audio 模型与 LLM 模型需使用同一版本（同为 E2B 或同为 E4B），不可混用。

## 7. 其他说明

本仓库默认使用以下镜像站点获取模型：
- [ModelScope](https://www.modelscope.cn)（国内推荐）
- [HuggingFace Mirror](https://hf-mirror.com)

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

- Qwen2.5
- Qwen2.5-VL
- Qwen2.5-Omni
- Qwen3
- Qwen3-VL
- Qwen3-Embedding
- FastVLM
- InternVLM
- Janus-Pro
- SmolVLM
- MiniCPM-V-4
- UI_TARS
- GME-Qwen2-VL
- glm-edge
- MobileNetV1/V2、ResNet50
- YOLOv5/v6/v8
- SenseVoice
- HY-MT1.5

## 2. 支持的平台

| 主芯片 | 协处理器 | 操作系统 |
|--------|----------|----------|
| RK3588 系列 | RK1820 / RK1828 | Linux / Android |
| RK3576 系列 | RK1820 / RK1828 | Linux / Android |

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
| `--quant` | bool | 启用 AWQ + GRQ 量化算法 | `False` |
| `--modelscope` | bool | 从 ModelScope 下载模型（国内用户推荐） | `False` |

> **说明**：
> - 使用 AWQ + GRQ 量化时，模型自带量化参数，转换 RKNN 时无需量化数据集
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
> - 使用 AWQ + GRQ 量化时，转换 RKNN 会忽略量化数据集
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
> - 使用 AWQ + GRQ 量化时，模型自带量化参数，转换 RKNN 时无需量化数据集
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
> - 使用 AWQ + GRQ 量化时，转换 RKNN 会忽略量化数据集
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

## 5. 模型适配指南

- **同系列模型兼容**：同一系列的示例程序相互兼容。例如，Qwen2.5-0.5B 的示例可直接用于 Qwen2.5-7B，只需修改模型加载路径。

- **新模型适配**：对于本仓库未收录的 LLM 模型，请参考 [LLM 模型适配教程](LLM_model_modification_guide_CN.md) 进行 ONNX 导出和部署移植。

## 6. 注意事项

- **Transformers 版本**：不同模型依赖的 `transformers` 版本可能不同，导出 ONNX 前请安装对应版本。版本信息可从模型的config.json（例如https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json ）中的 `transformers_version` 字段获取。

- **PyTorch 版本**：建议使用 PyTorch <= 2.8.0（Qwen3-VL等少数模型，需 PyTorch >= 2.9.0，具体看对应模型下的requirements.txt）

## 7. 其他说明

本仓库默认使用以下镜像站点获取模型：
- [ModelScope](https://www.modelscope.cn)（国内推荐）
- [HuggingFace Mirror](https://hf-mirror.com)

[中文](README_CN.md)

# RKNN3-MODEL-ZOO

RKNN3 SDK provides the complete software stack for deploying AI models on RK1820/RK1828 coprocessors, including:

- **[RKNN3-Toolkit](https://github.com/airockchip/rknn3-toolkit)**: PC-side software development kit for model conversion, inference, performance evaluation, etc.
- **RKNN3 Runtime**: On-board runtime library providing C/C++ programming interfaces for deploying RKNN models and accelerating AI applications.
- **[RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)**: Model conversion and deployment example repository, including reference implementations for CNN / LLM / VLM and other models.

This repository provides a complete model deployment workflow:
- **Model Export**: Export HuggingFace / PyTorch models to ONNX format
- **Model Conversion**: Convert ONNX models to RKNN format using RKNN3 Toolkit
- **On-board Deployment**: Provide C++ inference example code

## 1. Supported Models

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

## 2. Supported Platforms

| Host SoC | Coprocessor | OS |
|----------|-------------|-----|
| RK3588 Series | RK1820 / RK1828 | Linux / Android |
| RK3576 Series | RK1820 / RK1828 | Linux / Android |

## 3. LLM Model Deployment (Example: Qwen2.5-3B)

### 3.1 Export Model Configuration

> **Requirements**: Python 3.10

```bash
cd rknn3_model_zoo/
pip install -r requirements.txt
export PYTHONPATH=./

cd examples/Qwen2_5/python/
python export_llm.py --quant
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--load_weight` | bool | Load model weights (`False` exports structure only) | `True` |
| `--quan_dataset` | bool | Generate quantization dataset (requires `--load_weight=True`) | `True` |
| `--model_path` | str | Model path or HuggingFace name | `Qwen/Qwen2.5-3B-Instruct` |
| `--export_llm_path` | str | Export path | `../model/llm/Qwen2.5-3B-Instruct.onnx` |
| `--quant` | bool | Enable AWQ + GRQ quantization | `False` |
| `--modelscope` | bool | Download from ModelScope (recommended for China users) | `False` |

> **Notes**:
> - When using AWQ + GRQ quantization, the model contains quantization parameters; no quantization dataset is needed for RKNN conversion
> - Exported configuration files include: ONNX model, Config, Tokenizer, and Embed files

### 3.2 Export RKNN Model

```bash
python export_rknn.py
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--onnx_path` | str | Input ONNX model path | `../model/llm/Qwen2.5-3B-Instruct.onnx` |
| `--config` | str | Model Config path | `../model/llm/Qwen2.5-3B-Instruct.config.pkl` |
| `--rknn_path` | str | Output RKNN model path | `../model/llm/Qwen2.5-3B-Instruct.rknn` |
| `--dataset_path` | str | Quantization dataset path | `../data/dataset.txt` |

> **Notes**:
> - When using AWQ + GRQ quantization, RKNN conversion ignores the quantization dataset
> - Export uses weight-separated mode, generating both `.rknn` and `.weight` files

### 3.3 Linux Platform Example

#### Build

```bash
cd ../../../

# Optional: specify cross-compiler path
export GCC_COMPILER=<GCC_COMPILER_PATH>

# Build command
./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d Qwen2_5

# Example: RK3588 platform
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5
```

#### Push to Board

```bash
adb push install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_Qwen2_5_demo/ /data/
```

#### Run

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

## 4. VLM Model Deployment (Example: FastVLM-1.5B)

### 4.1 Export Model Configuration

> **Requirements**: Python 3.10

```bash
cd rknn3_model_zoo/
pip install -r requirements.txt
export PYTHONPATH=./

cd examples/FastVLM/python/

# Export LLM model configuration
cd ./llm
python export_llm.py --quant

# Export Vision model configuration
cd ../vision/
python export_vision.py
```

#### LLM Export Parameters (export_llm.py)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--load_weight` | bool | Load model weights (`False` exports structure only) | `True` |
| `--quan_dataset` | bool | Generate quantization dataset (requires `--load_weight=True`) | `True` |
| `--model_path` | str | Model path or HuggingFace name | `../../llava-fastvithd_1.5b_stage3` |
| `--export_llm_path` | str | Export path | `../../model/llm/FastVLM-llm.onnx` |
| `--quant` | bool | Enable AWQ + GRQ quantization | `False` |
| `--modelscope` | bool | Download from ModelScope (recommended for China users) | `False` |

> **Notes**:
> - Model needs to be downloaded externally
> - When using AWQ + GRQ quantization, the model contains quantization parameters; no quantization dataset is needed for RKNN conversion
> - Exported configuration files include: ONNX model, Config, Tokenizer, and Embed files

#### Vision Export Parameters (export_vision.py)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--load_weight` | bool | Load model weights | `True` |
| `--model_path` | str | Model path | `../../llava-fastvithd_1.5b_stage3` |
| `--export_vision_path` | str | Export path | `../../model/vision/FastVLM-vision.onnx` |
| `--img_size` | int | Input image resolution | `512` |

> **Note**: Vision model exports ONNX format only

### 4.2 Export RKNN Model

```bash
# Export Vision RKNN model
cd ./vision
python export_rknn.py

# Export LLM RKNN model
cd ../llm
python export_rknn.py
```

#### Vision RKNN Export Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--onnx_path` | str | Input ONNX model path | `../../model/vision/FastVLM-vision.onnx` |
| `--rknn_path` | str | Output RKNN model path | `../../model/vision/FastVLM-vision.rknn` |
| `--dataset_path` | str | Quantization dataset path | `../../../../datasets/MMBench/vision/datasets.txt` |

> **Note**: Uses Normal quantization by default; set `quantized_algorithm='grq'` for higher accuracy

#### LLM RKNN Export Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--onnx_path` | str | Input ONNX model path | `../../model/llm/FastVLM-llm.onnx` |
| `--config` | str | Model Config path | `../../model/llm/FastVLM-llm.config.pkl` |
| `--rknn_path` | str | Output RKNN model path | `../../model/llm/FastVLM-llm.rknn` |
| `--dataset_path` | str | Quantization dataset path | `../../data/llm/dataset.txt` |

> **Notes**:
> - When using AWQ + GRQ quantization, RKNN conversion ignores the quantization dataset
> - Export uses weight-separated mode, generating both `.rknn` and `.weight` files

### 4.3 Linux Platform Example

#### Build

```bash
cd rknn3_model_zoo/

# Optional: specify cross-compiler path
export GCC_COMPILER=<GCC_COMPILER_PATH>

# Build command
./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d FastVLM

# Example: RK3588 platform
./build-linux.sh -t rk3588 -a aarch64 -d FastVLM
```

#### Push to Board

```bash
adb push install/<TARGET_PLATFORM>_linux_<ARCH>/rknn_FastVLM_demo/ /data/
```

#### Run

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

## 5. Model Adaptation Guide

- **Same-series Compatibility**: Examples within the same model series are interchangeable. For instance, the Qwen2.5-0.5B example works directly with Qwen2.5-7B by simply changing the model loading path.

- **New Model Adaptation**: For LLM models not included in this repository, refer to the [LLM Model Adaptation Tutorial](LLM_model_modification_guide_EN.md) for ONNX export and deployment porting.

## 6. Important Notes

- **Transformers Version**: Different models may require different `transformers` versions. Before exporting to ONNX, install the correct version. Version info can be found in the `transformers_version` field of the model's config.json (e.g., https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json ).

- **PyTorch Version**: Recommended PyTorch <= 2.8.0 (For Qwen3-VL and a few other models, PyTorch >= 2.9.0 is required. Please refer to the requirements.txt under the corresponding model for specific details.)

## 7. Additional Notes

This repository uses the following mirror sites by default to obtain model files:
- [ModelScope](https://www.modelscope.cn) (recommended for China users)
- [HuggingFace Mirror](https://hf-mirror.com)

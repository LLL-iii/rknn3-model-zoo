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

### ASR (Speech Recognition)

- Qwen3-ASR (Streaming / Non-streaming)
- SenseVoiceSmall
- Zipformer

### TTS (Text-to-Speech)

- Qwen3-TTS
- VITS (LJSpeech / VCTK)

### Embedding / Reranker

- Qwen3-Embedding
- Qwen3-Reranker

### Translation

- HY-MT1.5

### OCR

- PaddleOCR-VL

### CV (Computer Vision)

- MobileNetV1 / MobileNetV2 / ResNet
- YOLOv5 / YOLOv6 / YOLOv8
- DINOv3
- SigLIP2

## 2. Supported Platforms

| Host SoC | Coprocessor | OS |
|----------|-------------|-----|
| RK3588 Series | RK1820 / RK1828 | Linux / Android |
| RK3576 Series | RK1820 / RK1828 | Linux / Android |
| RK3572 Series | - | Linux / Android |

> **Build and runtime library notes**:
> - The top-level build scripts `build-linux.sh` / `build-android.sh` accept `-t rk3588`, `rk3576`, `rk3572`, and `x86`.
> - RKNN3 runtime libraries installed into each demo's `lib/` directory are SoC-specific:
>   - `RK3588` / `RK3576`: `librknn3_api.so` and `librknn3_api_rkcp.so`
>   - `RK3572`: `librknn3_api.so` and `librknn3_api_native.so`

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
| `--quant` | bool | Enable GRQ quantization | `False` |
| `--modelscope` | bool | Download from ModelScope (recommended for China users) | `False` |

> **Notes**:
> - When using GRQ quantization, the model contains quantization parameters; no quantization dataset is needed for RKNN conversion
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
> - When using GRQ quantization, RKNN conversion ignores the quantization dataset
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

### 4.4 SpeedUP Usage

The Qwen2.5-VL and Qwen3-VL examples can link against the SpeedUP third-party library for inference acceleration.

#### File Location

Keep the following files in the release package:

```text
3rdparty/SpeedUP/
├── include/speedup.h
├── Linux/aarch64/libSpeedUP.so
└── Android/arm64-v8a/libSpeedUP.so
```

#### Build

Qwen2.5-VL:

```bash
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5_VL
```

Qwen3-VL:

```bash
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL
```

After installation, `libSpeedUP.so` is copied into the corresponding demo `lib/` directory.

#### Runtime Arguments

Qwen2.5-VL:

```bash
./rknn_qwen2_5_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

Qwen3-VL:

```bash
./rknn_qwen3_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` is optional:

| Value | Mode |
|-------|------|
| `1.0` | Auto |
| `0.0` | Disabled |
| `(0.0, 1.0)` | Manual |

For RKNN3 multi-core devices, this is usually suitable:

```bash
0xff 0xff
```

#### Examples

Qwen2.5-VL:

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
    "Describe this image" \
    392 392 \
    1.0
```

Qwen3-VL:

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
    "Describe this image" \
    384 384 \
    1.0
```

## 5. Model Adaptation Guide

- **Same-series Compatibility**: Examples within the same model series are interchangeable. For instance, the Qwen2.5-0.5B example works directly with Qwen2.5-7B by simply changing the model loading path.

- **New Model Adaptation**: For LLM models not included in this repository, refer to the [LLM Model Adaptation Tutorial](LLM_model_modification_guide_EN.md) for ONNX export and deployment porting.

## 6. Important Notes

- **Transformers Version**: Different models may require different `transformers` versions. Before exporting to ONNX, install the correct version. Version info can be found in the `transformers_version` field of the model's config.json (e.g., https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json ). Some models have special version requirements; refer to the `requirements.txt` in each example directory.

- **PyTorch Version**: Recommended PyTorch <= 2.8.0 (Qwen3-VL, Gemma-4 service models require PyTorch >= 2.9.0; PaddleOCR-VL requires transformers == 4.55.0. Please refer to the requirements.txt under the corresponding model for specific details.)

- **Module Compatibility**: Gemma-4 Audio and LLM models must use the same version (both E2B or both E4B); mixing versions is not supported.

## 7. Additional Notes

This repository uses the following mirror sites by default to obtain model files:
- [ModelScope](https://www.modelscope.cn) (recommended for China users)
- [HuggingFace Mirror](https://hf-mirror.com)

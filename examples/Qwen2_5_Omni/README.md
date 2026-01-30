# Qwen2.5-Omni 模型部署说明

模型地址：[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)

## 1. 部署环境

本仓库 `requirements.txt` 中的 `transformers==4.51.3` 无法导出该模型，请使用以下命令安装特定版本：

```bash
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
```

> ⚠️ 还需安装最新版 rknn3-toolkit

## 2. 导出 RKNN 模型

### 2.1 Vision 模型

```bash
cd python/vision

# 导出 ONNX 模型
python export_vision.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

### 2.2 Audio 模型

```bash
cd python/audio

# 导出 ONNX 模型
python export_audio.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

### 2.3 LLM 模型

```bash
cd python/llm

# 导出 ONNX 模型
python export_llm.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

> 导出完成后，模型文件将生成在 `model` 目录下。

## 3. C++ 板端部署

### 3.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5_Omni
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Qwen2_5_Omni_demo/` 目录：

```
rknn_Qwen2_5_Omni_demo/
├── lib/
│   ├── librga.so
│   └── librknn3_api.so
├── model/
│   ├── Qwen2.5-Omni-3B-audio.rknn
│   ├── Qwen2.5-Omni-3B-audio.weight
│   ├── Qwen2.5-Omni-3B-llm.rknn
│   ├── Qwen2.5-Omni-3B-llm.weight
│   ├── Qwen2.5-Omni-3B-llm.embed.bin
│   ├── Qwen2.5-Omni-3B-llm.tokenizer.gguf
│   ├── Qwen2.5-Omni-3B-vision.rknn
│   └── Qwen2.5-Omni-3B-vision.weight
├── mel_128_filters.txt
├── demo.jpg
├── demo.wav
└── rknn_qwen2_5_omni_demo
```

### 3.2 部署到开发板

```bash
# 推送 demo 目录
adb push rknn_Qwen2_5_Omni_demo /data/

# 推送运行库
adb push rknn_Qwen2_5_Omni_demo/lib/* /usr/lib/
```

### 3.3 运行示例

```bash
adb shell
cd /data/rknn_Qwen2_5_Omni_demo
```

#### 示例 1：Vision + LLM（图像理解）

```bash
./rknn_qwen2_5_omni_demo \
    model/Qwen2.5-Omni-3B-vision.rknn model/Qwen2.5-Omni-3B-vision.weight \
    model/Qwen2.5-Omni-3B-audio.rknn model/Qwen2.5-Omni-3B-audio.weight \
    model/Qwen2.5-Omni-3B-llm.rknn model/Qwen2.5-Omni-3B-llm.weight \
    model/Qwen2.5-Omni-3B-llm.tokenizer.gguf model/Qwen2.5-Omni-3B-llm.embed.bin \
    0xff 0xff 0xff \
    demo.jpg demo.wav \
    "<image>描述下这张图："
```

输出示例：
```
这张图片展示了一个穿着宇航服的宇航员在月球表面。他手里拿着一瓶绿色啤酒，旁边有一个装满饮料的冷藏箱。背景是地球和星空，给人一种太空旅行的感觉。
```

#### 示例 2：Audio + LLM（语音转文本）

```bash
./rknn_qwen2_5_omni_demo \
    model/Qwen2.5-Omni-3B-vision.rknn model/Qwen2.5-Omni-3B-vision.weight \
    model/Qwen2.5-Omni-3B-audio.rknn model/Qwen2.5-Omni-3B-audio.weight \
    model/Qwen2.5-Omni-3B-llm.rknn model/Qwen2.5-Omni-3B-llm.weight \
    model/Qwen2.5-Omni-3B-llm.tokenizer.gguf model/Qwen2.5-Omni-3B-llm.embed.bin \
    0xff 0xff 0xff \
    demo.jpg demo.wav \
    "<audio>将这段语音转为文本."
```

输出示例：
```
这段语音的文本内容是：'图片里是什么'
```

#### 示例 3：Vision + Audio + LLM（多模态理解）

```bash
./rknn_qwen2_5_omni_demo \
    model/Qwen2.5-Omni-3B-vision.rknn model/Qwen2.5-Omni-3B-vision.weight \
    model/Qwen2.5-Omni-3B-audio.rknn model/Qwen2.5-Omni-3B-audio.weight \
    model/Qwen2.5-Omni-3B-llm.rknn model/Qwen2.5-Omni-3B-llm.weight \
    model/Qwen2.5-Omni-3B-llm.tokenizer.gguf model/Qwen2.5-Omni-3B-llm.embed.bin \
    0xff 0xff 0xff \
    demo.jpg demo.wav \
    "<image><audio>"
```

输出示例：
```
这张图片展示了一个穿着宇航服的人，站在月球表面。他手里拿着一个绿色的瓶子和一个保温箱，背景是地球在夜空中。这个场景显然是经过艺术处理或创意设计的，并非真实拍摄的照片。
```

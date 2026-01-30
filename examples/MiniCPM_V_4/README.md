# MiniCPM-V-4 模型部署说明

模型地址：[openbmb/MiniCPM-V-4](https://huggingface.co/openbmb/MiniCPM-V-4)

## 1. 部署环境

> ⚠️ 需安装最新版 rknn3-toolkit

## 2. 导出 RKNN 模型

### 2.1 Vision 模型

```bash
cd python/vision

# 导出 ONNX 模型
python export_vision.py --modelscope

# 导出 RKNN 模型
python export_rknn.py
```

### 2.2 LLM 模型

```bash
cd python/llm

# 导出 ONNX 模型
python export_llm.py --modelscope --quant

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
./build-linux.sh -t rk3588 -a aarch64 -d MiniCPM_V_4
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_MiniCPM_V_4_demo/` 目录：

```
rknn_MiniCPM_V_4_demo/
├── lib
│   ├── librga.so
│   └── librknn3_api.so
├── model
│   ├── MiniCPM-V-4-llm.embed.bin
│   ├── MiniCPM-V-4-llm.rknn
│   ├── MiniCPM-V-4-llm.tokenizer.gguf
│   ├── MiniCPM-V-4-llm.weight
│   ├── MiniCPM-V-4-vision.rknn
│   ├── MiniCPM-V-4-vision.weight
│   └── demo.jpg
└── rknn_minicpmv4_demo
```

### 3.2 部署到开发板

```bash
# 推送 demo 目录
adb push rknn_MiniCPM_V_4_demo /data/

# 推送运行库
adb push rknn_MiniCPM_V_4_demo/lib/* /usr/lib/
```

### 3.3 运行示例

```bash
adb shell
cd /data/rknn_MiniCPM_V_4_demo

./rknn_minicpmv4_demo \
    model/MiniCPM-V-4-vision.rknn model/MiniCPM-V-4-vision.weight \
    model/MiniCPM-V-4-llm.rknn model/MiniCPM-V-4-llm.weight \
    model/MiniCPM-V-4-llm.tokenizer.gguf model/MiniCPM-V-4-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "描述下这张图."
```

输出示例：
```
这幅画描绘了一个宇航员在月球表面放松的场景。宇航员的头盔反射着周围环境，表明了太空环境的特征。地球清晰可见，突出了人类探索宇宙的壮丽背景。宇航员手持一瓶绿色玻璃瓶，可能是一罐饮料，暗示了一种轻松或庆祝的时刻。绿色垃圾桶的存在以及梯子的存在增加了这个场景的真实性，因为它们是月球基地中常见的设备。这幅画捕捉了太空探索中的幽默和异想天开的一面，将严肃的科学任务与日常的休闲活动相结合。
```

# MiniCPM-V-4 Model Deployment Instructions

Model Address: https://huggingface.co/openbmb/MiniCPM-V-4

## 1. Deployment Environment

> ⚠️ Requires installing the latest version of rknn3-toolkit

## 2. Export RKNN Model

### 2.1 Vision Model

```bash
cd python/vision

# Export ONNX model
python export_vision.py --modelscope

# Export RKNN model
python export_rknn.py
```

### 2.2 LLM Model

```bash
cd python/llm

# Export ONNX model
python export_llm.py --modelscope --quant

# Export RKNN model
python export_rknn.py
```

> After the export is completed, the model files will be generated in the `model` directory.

## 3. C++ Deployment

### 3.1 Compilation

```bash
cd rknn3_model_zoo/

# Set cross-compilation toolchain
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# Compile
./build-linux.sh -t rk3588 -a aarch64 -d MiniCPM_V_4
```

After compilation, the files are generated in the `install/rk3588_linux_aarch64/rknn_MiniCPM_V_4_demo/` directory:

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

### 3.2 Deploy to Development Board

```bash
# Push the demo directory
adb push rknn_MiniCPM_V_4_demo /data/

# Push runtime libraries
adb push rknn_MiniCPM_V_4_demo/lib/* /usr/lib/
```

### 3.3 Run the Example

```bash
adb shell
cd /data/rknn_MiniCPM_V_4_demo

./rknn_minicpmv4_demo \
    model/MiniCPM-V-4-vision.rknn model/MiniCPM-V-4-vision.weight \
    model/MiniCPM-V-4-llm.rknn model/MiniCPM-V-4-llm.weight \
    model/MiniCPM-V-4-llm.tokenizer.gguf model/MiniCPM-V-4-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "Please describe the content of the picture."
```

Example output:
```
The image depicts an astronaut in a white space suit with gold-tinted visor, sitting on the surface of what appears to be another planet or moon. The background showcases Earth prominently visible against deep blackness filled by stars and cosmic dust clouds typical for outer solar system views from celestial bodies other than our home world (Earth) itself which is often referred as a "blue marble" due to its blue oceans contrasting with the green landmasses seen in satellite imagery.

The astronaut's posture suggests relaxation or leisure, indicated by their legs being stretched out and crossed at an angle while holding up what looks like either part of space equipment (possibly gloves) on one hand but more prominently a bottle that resembles typical human beverage containers used for beer bottles due to its green color. This implies the astronaut might be taking time off from work or engaging in recreational activities during their mission, which is quite unusual given usual expectations around such environments where hydration and sustenance are critical needs rather than leisurely pursuits like drinking beverages while on duty outside Earth's atmosphere under zero gravity conditions unless explicitly stated otherwise.

In the foreground near this astronaut figure lies a green object that resembles an old-fashioned cooler or storage container, commonly used for keeping food and drinks cold during space missions due to its insulated properties designed specifically against temperature fluctuations in microgravity environments encountered beyond Earth's atmosphere where conventional household items may not function as intended.

To the right side of this scene there is a metallic ladder structure protruding from behind some rocky terrain, possibly indicating access points or platforms used for movement between different areas on such extraterrestrial surfaces like lunar landscapes often depicted in science fiction media portrayals involving space exploration endeavors beyond our planetary confines.

Overall the image conveys an intriguing blend of human leisure amidst extraordinary circumstances while simultaneously highlighting technological advancements necessary to sustain life and operations within extreme environments far removed from Earth's familiar gravity-bound conditions, thus inviting viewers into a thought experiment about what it might be like for astronauts when they venture beyond our planetary boundaries.

```
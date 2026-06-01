# PaddleOCR-VL Model Deployment Guide

## 1. Environment Requirements

PaddleOCR-VL has specific dependency requirements that are incompatible with the `requirements.txt` in this repository. Please install the following dependencies manually:

```
transformers == 4.55.0
```

> ⚠️ Using the default dependencies will cause model conversion to fail. Please install the versions specified above.

## 2. Model Pruning Strategy

To support larger context lengths, appropriate pruning is required when deploying large-scale multimodal models.

### 2.1 Vision Model Pruning

Some operators are offloaded to the CPU of the host device (e.g., RK3588).

### 2.2 MLP-AR Model Partitioning

Due to dynamic shape transformations involved in the MLP-AR layer, it is partitioned into a separate subgraph. Shape manipulation operations are executed on the CPU.

## 3. Model Export

```bash
# 导出 LLM ONNX 模型
python export_llm.py \
    --model_path PaddleOCR-VL/PaddleOCR-VL-0.9B \
    --export_llm_path ../../model/llm/PaddleOCR-llm.onnx

# 导出 LLM RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/PaddleOCR-llm.onnx \
    --config ../../model/llm/PaddleOCR-llm.config.pkl \
    --rknn_path ../../model/llm/PaddleOCR-llm.rknn

# 导出 Vision ONNX 模型 和 MLP-AR ONNX 模型
python export_vision.py \
    --model_path PaddleOCR-VL/PaddleOCR-VL-0.9B \
    --export_vision_path ../../model/vision/PaddleOCR-vision.onnx \
    --export_mlp_AR_path ../../model/vision/PaddleOCR-vision-mlp_AR.onnx

# 导出 Vision RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/vision/PaddleOCR-vision.onnx \
    --rknn_path ../../model/vision/PaddleOCR-vision.rknn \
    --mlpar_onnx_path ../../model/vision/PaddleOCR-vision-mlp_AR.onnx \
    --mlpar_rknn_path ../../model/vision/PaddleOCR-vision-mlp_AR.rknn
```

## 4. KV Cache INT4 Quantization

In large-scale language model inference, a KV (Key/Value Cache) is used to store historical attention keys to avoid redundant calculations and thus improve inference speed. As sequence length increases, the memory footprint of the KV Cache increases rapidly. To reduce the storage bandwidth and memory access overhead of the KV Cache, quantization can be used to convert it from FP16/FP32 to INT8 or a lower bit width representation. However, since the KV Cache value distribution changes dynamically token by token over time, using a uniform quantization parameter for the entire KV segment can lead to accumulated quantization errors, affecting inference accuracy. Therefore, group quantization is typically used to reduce accuracy loss.

Currently, RKNN's LLM supports two KV cache quantization modes:

Int8_to_F16 (default): Stores in INT8 format, converts back to FP16 during computation;

Int4_to_F16 (suitable for longer contexts, with some precision loss): Stores in INT4 format, converts back to FP16 during computation.

For support of longer context lengths and further compression of KV cache memory, it is recommended to enable the Int4_to_F16 mode.

The configuration for enabling Int4_to_F16 RKNN model transformation is as follows:

```python
rknn.config(target_platform='rk1820', 
          quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',
          max_ctx_len           =2048,
          max_position_embeddings=2048,
          kvcache_store_method='GroupQuant', kvcache_dtype='Int4_to_F16', 
          kvcache_group_size=16, kvcache_residual_depth=64,
          )
```
- Note: The above configuration is located in the python/llm/export_rknn.py file. Please adjust the relevant parameters according to your actual needs.

## 5. Vision Model Resolution Adjustment

Use `--img_h` and `--img_w` parameters to adjust input resolution (must be a multiple of 28):

```bash
python export_vision.py --img_h 504 --img_w 504
```

> ⚠️ **Note**:
> - Higher resolution increases memory usage and affects the maximum LLM context length
> - Some resolutions may be incompatible with the RKNN inference framework; contact the RKNPU team if errors occur

## 6. C++ Deployment

If you modified the Vision model resolution, update the parameters in `rknn_paddleocr_vl_vision.h`:

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

### 1. build c++ demo

c++ demo code：`examples/paddleocr_vl/cpp/`

```bash
cd ../../../../
./build-linux.sh -t rk3588 -a aarch64 -d paddleocr_vl
```

building success, output directory：`install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo`, including model files, dependency libraries and demo programs. The directory structure is as follows:

```bash
- install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo
 - lib
   - librknn3_api.so
   - librga.so
 - model
   - PaddleOCR-vision.rknn
   - PaddleOCR-vision.weight
   - PaddleOCR-vision-mlp_AR.rknn
   - PaddleOCR-vision-mlp_AR.weight
   - position_embedding_model.bin
   - PaddleOCR-llm.rknn
   - PaddleOCR-llm.weight
   - PaddleOCR-llm.tokenizer.gguf
   - PaddleOCR-llm.embed.bin
   - test.png
 - rknn_paddleocr_vl_demo
```

### 2. push demo program to RK3588 board

```bash
adb push install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo /userdata/
```

### 3. run c++ demo

```bash
adb shell
cd /userdata/rknn_paddleocr_vl_demo
./rknn_paddleocr_vl_demo model/PaddleOCR-vision.rknn model/PaddleOCR-vision.weight model/position_embedding_model.bin model/PaddleOCR-llm.rknn model/PaddleOCR-llm.weight model/PaddleOCR-llm.tokenizer.gguf model/PaddleOCR-llm.embed.bin model/PaddleOCR-vision-mlp_AR.rknn model/PaddleOCR-vision-mlp_AR.weight 0xff 0xff 0xff model/test.png "table"
```

if run success, will print inference result and performance, as follows:

```bash
--> inference paddleocr_vl llm model
rknn_session_run
<fcel>考评项目<lcel><fcel>权重<fcel>考评内容<fcel>评分标准<fcel>评分方法<fcel>自评得分<nl><fcel>一、组织领导体系(120分)<fcel>学校安全工作目标责任制<fcel>20<fcel>建立健全创造平安校园组织领导工作机制， 责任明确、措施落实<fcel>学校有创建平安校园工作组织领导机构，第一责任人到位，安全岗位职官明确的得20分<fcel>查看会议记录、学校文件、岗位安全职责分工等资料<ecel><nl><ucel><ucel><fcel>20<fcel>学校有创建平安校园工作年度目标，计划<fcel>每当年创建目标、计划的约20分； 创建目标和计划重点不突出、针对性不强的扣10分；无创建目标和计划的不得分<fcel>查看台账资料<ecel><nl><ucel><ucel><fcel>30<fcel>定期研究分析校园安全稳定问题，提出针对性对策措施<fcel>每月不少于1次安全分析会议（校园平安情况排查分析研究部署）等20分；有针对突出措施的得10分；无会议记录扣10分<fcel>查看会议、工作记录<ecel><nl><ucel><fcel>各部门安全工作责任分解<fcel>30<fcel>学校签订岗位安全工作任务书<fcel>学校根据各部门特点制定相应的责任书，签订、落实并上交责任书（校园安全工作责任书）等20分；部门岗位有任书没有针对性的扣10分；不完全落实扣15分；未落实不得分<fcel>查看责任书<ecel><nl><ucel><fcel>安全保卫人员配备<fcel>20<fcel>校园安全管理机构、安全保卫人员落实<fcel>岗具体负责学校安全工作的职能科室（安管办）和专职的安全保卫人员得20分；兼职的得10分；没有配备的不得分<fcel>查看教职工注册等资料，实地考察<ecel><nl><fcel>二、制度建设体系(150分)<fcel>治安制度<fcel>25<fcel>有门卫、巡逻、实验室、重点部位、场所、学生生活区安全保卫制度<fcel>有制度得25分；制度不全扣10分<fcel>查看制度<ecel><nl>

--------------------Finished-------------------- 

--------------------------------------------------------------------------------------
 Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second      
--------------------------------------------------------------------------------------
 Prefill       58.99            338       0.17                     5729.88                
 Generate      1786.05          423       4.22                     236.84                 
--------------------------------------------------------------------------------------
 Vision latency = 520.81 ms, FPS = 1.92
```
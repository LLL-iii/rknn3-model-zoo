# PaddleOCR-VL 模型部署说明

## 1. 部署环境

由于 PaddleOCR-VL 模型对依赖库版本有特殊要求，与本仓库 `requirements.txt` 中的版本不兼容，请手动安装以下依赖：

```
transformers == 4.55.0
```

> ⚠️ 使用默认依赖会导致模型转换失败，请务必按上述版本安装。

## 2. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时可进行适当裁剪。

### 2.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 2.2 MLP-AR 模型拆分

由于 MLP-AR 层涉及动态形状变换，将其拆分为单独子图。形状变化操作在 CPU 上运行。

## 3. 导出模型流程

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

## 4. KV Cache INT4 量化
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


## 5. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 28 的倍数）：

```bash
python export_vision.py --img_h 504 --img_w 504
```

> ⚠️ **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

## 6. C++ 部署说明

若修改了 Vision 模型的分辨率，需同步调整 `rknn_paddleocr_vl_vision.h` 中的参数：

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

### 1. 编译c++ demo

c++ demo代码位置：`examples/paddleocr_vl/cpp/`

```bash
cd ../../../../
./build-linux.sh -t rk3588 -a aarch64 -d paddleocr_vl
```

编译成功后，c++ demo程序保存在`install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo`下，包括模型文件、依赖库文件及demo程序。目录结构如下：

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

### 2. 推送demo程序到RK3588开发板

```bash
adb push install/rk3588_linux_aarch64/rknn_paddleocr_vl_demo /userdata/
```

### 3. 运行c++ demo程序

```bash
adb shell
cd /userdata/rknn_paddleocr_vl_demo
./rknn_paddleocr_vl_demo model/PaddleOCR-vision.rknn model/PaddleOCR-vision.weight model/position_embedding_model.bin model/PaddleOCR-llm.rknn model/PaddleOCR-llm.weight model/PaddleOCR-llm.tokenizer.gguf model/PaddleOCR-llm.embed.bin model/PaddleOCR-vision-mlp_AR.rknn model/PaddleOCR-vision-mlp_AR.weight 0xff 0xff 0xff model/test.png "table"
```

运行成功后，会打印出识别结果及推理性能，如下：

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
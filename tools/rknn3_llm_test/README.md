
# rknn3_llm_test 说明文档

该工具用于评估大语言模型(LLM)在数据集上的精度和性能表现:

- 支持的数据集:
  - CMMLU: 中文多任务语言理解评测基准

- 性能评估指标:
  - TTFT (Time To First Token): 首个 token 生成延迟
  - Prefill: Prefill 阶段推理速度
  - TPOT (Time Per Output Token): Docode 阶段每个输出 token 的平均生成时间
  - TPS (Tokens Per Second): Docode 阶段每秒生成的 token 数量

## Linux平台使用示例

### 编译

```bash
# 请先指定编译器路径
(optional)export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> [-b <build_type>]

# 例如
./build-linux.sh -t rk3588 -a aarch64 -b Release
```

### 推送到板端

```shell
adb push install/rknn3_llm_test_<TARGET_PLATFORM>_Linux/ /data/
```

### 运行

```sh
adb shell
cd /data/rknn3_llm_test_<TARGET_PLATFORM>_Linux

export LD_LIBRARY_PATH=./lib

# cmmlu数据集测试命令
# Usage: ./rknn3_session_test_cmmlu <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask>
./rknn3_session_test_cmmlu model/Qwen2.5-0.5B-Instruct.rknn model/Qwen2.5-0.5B-Instruct.weight model/Qwen2.5-0.5B-Instruct.tokenizer.gguf model/Qwen2.5-0.5B-Instruct.embed.bin 1024 1 0xff

# 性能测试命令，由于第一次会话有初始化耗时，所以性能数据需要看第二次会话 (当前输入为随机数据，其输出没有参考意义，只需关注性能数据)
# Usage: ./rknn3_session_test_eval_perf <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <n_input_tokens> <max_new_tokens> <core_mask>
./rknn3_session_test_eval_perf model/Qwen2.5-0.5B-Instruct.rknn model/Qwen2.5-0.5B-Instruct.weight model/Qwen2.5-0.5B-Instruct.tokenizer.gguf model/Qwen2.5-0.5B-Instruct.embed.bin 1024 128 128 0xff

```

参数说明: 
- `rknn_path`: rknn 文件路径
- `weight_path`: weight 文件路径
- `tokenizer.gguf`: tokenizer.gguf 文件路径
- `embedding.bin`: embedding.bin 文件路径
- `max_context_len`: 模型转换时 `rknn.config` 中配置的 max_ctx_len 值
- `n_input_tokens`: 模型输入的 token 数
- `max_new_tokens`: 每次会话最多生成的 token 数
- `core_mask`: 目前有 8 个核，对应 8 bit 数，使用哪一个核，就将哪一位置 1，例如使用核 0 和核 1，就将第 0 位和第 1 位置 1，得到的二进制数是0b11，对应的十六进制数是 0x3，core_mask 设置成 0x3

## Android平台使用示例

### 编译

```bash
# 请先指定编译器路径
(optional)export ANDROID_NDK_PATH=<ANDROID_NDK_PATH>

./build-android.sh -t <TARGET_PLATFORM> -a <ARCH> [-b <build_type>]

# 例如
./build-android.sh -t rk3588 -a arm64-v8a -b Release
```

### 推送到板端

```shell
adb root
adb remount
adb push install/rknn3_llm_test_<TARGET_PLATFORM>_Android/ /data/
```

### 运行

```sh
adb shell
cd /data/rknn3_llm_test_<TARGET_PLATFORM>_Android

export LD_LIBRARY_PATH=./lib

# cmmlu数据集测试命令
# Usage: ./rknn3_session_test_cmmlu <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask>
./rknn3_session_test_cmmlu model/Qwen2.5-0.5B-Instruct.rknn model/Qwen2.5-0.5B-Instruct.weight model/Qwen2.5-0.5B-Instruct.tokenizer.gguf model/Qwen2.5-0.5B-Instruct.embed.bin 1024 1 0xff

# 性能测试命令，由于第一次会话有初始化耗时，所以性能数据需要看第二次会话 (当前输入为随机数据，其输出没有参考意义，只需关注性能数据)
# Usage: ./rknn3_session_test_eval_perf <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <n_input_tokens> <max_new_tokens> <core_mask>
./rknn3_session_test_eval_perf model/Qwen2.5-0.5B-Instruct.rknn model/Qwen2.5-0.5B-Instruct.weight model/Qwen2.5-0.5B-Instruct.tokenizer.gguf model/Qwen2.5-0.5B-Instruct.embed.bin 1024 128 128 0xff

```

参数说明: 
- `rknn_path`: rknn 文件路径
- `weight_path`: weight 文件路径
- `tokenizer.gguf`: tokenizer.gguf 文件路径
- `embedding.bin`: embedding.bin 文件路径
- `max_context_len`: 模型转换时 `rknn.config` 中配置的 max_ctx_len 值
- `n_input_tokens`: 模型输入的 token 数
- `max_new_tokens`: 每次会话最多生成的 token 数
- `core_mask`: 目前有 8 个核，对应 8 bit 数，使用哪一个核，就将哪一位置 1，例如使用核 0 和核 1，就将第 0 位和第 1 位置 1，得到的二进制数是0b11，对应的十六进制数是 0x3，core_mask 设置成 0x3


## 参考结果
- 数据集精度评估结果: 以 `cmmlu` 数据集测试结果为例，最终评估结果如下
```sh
# 打印进度统计：当前测试数量/总测试数量，正确样本的个数，准确率百分比
Progress: 11582/11582 | Correct: 5801 | Accuracy: 50.09%
```

- 性能评估指标: 以 `Qwen2.5-0.5B-Instruct` 模型为例，性能数据如下
```sh
--------------------------------------------------------------------------------------
 Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second      
--------------------------------------------------------------------------------------
 Prefill       28.935           128       0.23                     4423.708                
 Generate      840.089          127       6.61                     151.190                                 
--------------------------------------------------------------------------------------
```
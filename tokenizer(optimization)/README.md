# RKNN3 Tokenizer

基于 Meta [meta-pytorch/tokenizers](https://github.com/pytorch/executorch/tree/main/third-party/tokenizers)（BSD-3 许可证）的轻量级 C++ 分词器。原始工程依赖 C++20（`<filesystem>`、`<string_view>` 等），本仓库已完成 C++17 降级改造，可直接用 GCC 7.4 / Clang 12 编译并部署到嵌入式板端。

**特性**：
- 直接加载 HuggingFace 模型目录（`tokenizer.json` / `tokenizer.model`），零格式转换
- 支持 BPE (ByteLevel) 和 SentencePiece 双后端，自动检测
- 单头文件 API，PIMPL 封装，易于集成
- 合并静态库 **~2.6 MB**（x86_64），无运行时依赖
- 峰值内存 **~28 MB**（Qwen3），手写 JSON 扫描器替代 DOM 树，较上游降 ~65%
- 端到端精度 100%（20 模型 × 103 深度测试 case）
- 跨平台：Linux x86_64 / aarch64 / Android arm64-v8a / Windows (Cygwin)

---

## 1. 编译

### 1.1 环境要求

| 工具 | 版本 |
|------|------|
| CMake | ≥ 3.18 |
| GCC / Clang | ≥ 7（支持 C++17） |
| PCRE2 | ≥ 10.40（静态编译进库） |

### 1.2 编译命令

```bash
# Linux x86_64（GCC 13 / C++17）
bash build.sh -s linux -a x86 -b Release

# Linux aarch64（GCC 7.4 Linaro / C++17）
# 默认使用 /opt/toolchains/gcc-linaro-7.4.1-2019.02-x86_64_aarch64-linux-gnu
bash build.sh -s linux -a aarch64 -b Release

# Android arm64-v8a（NDK r23c Clang 12 / C++17）
bash build.sh -s android -a arm64-v8a -b Release

# Windows（Cygwin 环境）
bash build.sh -s cygwin -a x86 -b Release
```

### 1.3 安装目录

编译后在 `install/tokenizer_<platform>/` 下生成：

```
install/tokenizer_linux_aarch64/
├── include/
│   └── Tokenizer.h          ← 唯一头文件
├── lib/
│   └── libtokenizer.a       ← 合并静态库 (~2.6 MB x86_64, ~5 MB aarch64)
└── demo/
    └── tokenize_demo        ← 命令行分词工具（已 strip）
```

### 1.4 交叉编译 aarch64

需要 ARM GCC 7.4 工具链（Linaro）：

```bash
# 修改 env_linux.sh 中的工具链路径
RK_AARCH64_TOOLCHAIN=/opt/toolchains/gcc-linaro-7.4.1-2019.02-x86_64-aarch64-linux-gnu/bin/
```

```bash
bash build.sh -s linux -a aarch64 -b Release
file install/tokenizer_linux_aarch64/demo/tokenize_demo
# 应输出: ELF 64-bit LSB executable, ARM aarch64
```

### 1.5 Android 编译

设置 NDK 路径：

```bash
# env_android.sh
ANDROID_NDK_PATH=/opt/toolchains/android-ndk-r23c
```

```bash
bash build.sh -s android -a arm64-v8a -b Release
ls install/tokenizer_android_arm64-v8a/demo/tokenize_demo
```

---

## 2. C++20 → C++17 降级说明

原始 meta-pytorch/tokenizers 要求 C++20。为了兼容 Linaro GCC 7.4（RK 芯片官方工具链）和 Android NDK r23c Clang 12，做了以下改造：

| 原始（C++20） | 改造后（C++17） | 涉及文件 |
|:---|:---|:---|
| `<filesystem>` / `std::filesystem::path` | POSIX `stat()` / `access()` / 字符串拼接 | `Tokenizer.cpp`, `hf_tokenizer.cpp` |
| `std::atomic<T> x = val;` | `std::atomic<T> x{val};` | `sentencepiece/src/util.cc` |
| C++17 `_v` 别名 / structured binding 保留 | （GCC 7.4 原生支持） | 无需改动 |
| `<string_view>`, `<optional>` | 标准 C++17 头文件（GCC 7.4 / Clang 12 均已自带） | 无需改动 |
| `CMAKE_CXX_STANDARD 20` | `CMAKE_CXX_STANDARD 17` | 全部 CMakeLists.txt |

**关键要点**：
- GCC 7.4 完全支持 C++17 的 `<string_view>`、`<optional>`、`_v` 别名、structured binding
- 唯一不支持的是 `<filesystem>`（GCC 8+），已用 POSIX 系统调用替代
- 不需要任何 compat shim 或 polyfill

---

## 3. 体积优化

### 3.1 x86_64（GCC 13）体积

| 优化手段 | 效果 |
|:---|:---|
| `-Os -ffunction-sections -fdata-sections` | 基线 ~6 MB → ~4.5 MB |
| `merge_libs.sh` 中 strip `-S` 所有 .o 文件 | 去掉 debug section |
| 关闭 LTO（`CMAKE_INTERPROCEDURAL_OPTIMIZATION`） | 避免 GIMPLE IR 膨胀（27 MB → 2.5 MB） |
| `-static-libstdc++ -static-libgcc` | 嵌入式兼容，不增加 .a 体积 |
| 合并前清理 Abseil .a 文件 | 消除 ~11 MB dead code |
| `merge_libs.sh` 最终 `strip -s` 合并库 | 去掉符号表 |
| **最终体积** | **~2.53 MB** |

### 3.2 aarch64（GCC 7.4）体积

aarch64 版本约 **5.28 MB**，约为 x86_64 的 2 倍，原因：
- GCC 7.4 代码生成密度低于 GCC 13（约 1.5x）
- ARM64 指令编码密度低于 x86_64（约 0.77x）
- 两者叠加约 2x 膨胀

如需缩小 aarch64 体积，可切换到 ARM GCC 13.3 工具链（需确认目标板端 GLIBC 版本兼容）。

### 3.3 合并脚本（merge_libs.sh）

```bash
# 解包 libtokenizer.a + libtokenizers.a + libsentencepiece.a + libre2.a
# 不拆 Abseil（SPM 用内部桩，Abseil 是死代码）
# strip -S 所有 .o（删除 .debug_* 段）
# ar -qcs 合并 + strip -s（删除符号表）
#
# 最终产出一个 libtokenizer_merged.a
```

---

## 4. API

### 4.1 构造函数

```cpp
/**
 * @brief 从 HuggingFace 模型目录加载
 * @param model_dir 包含 tokenizer.json（或 tokenizer.model）的目录路径
 *
 * 自动检测：目录中若存在 tokenizer.model，走 SentencePiece 后端；
 * 否则走 HFTokenizer（BPE / tokenizer.json）。
 */
Tokenizer(const char* model_dir);

/**
 * @brief 从内存加载 tokenizer.json
 * @param json_data tokenizer.json 的字节内容
 * @param json_size 字节数
 */
Tokenizer(const void* json_data, size_t json_size);
```

### 4.2 分词（Tokenize）

```cpp
/**
 * @brief 将输入文本编码为 token ID 序列
 * @param text        输入文本指针
 * @param text_len    输入文本长度（字节）
 * @param tokens      输出 token ID 数组
 * @param n_tokens_max 输出数组最大容量
 * @return 实际 token 数量，失败返回 -1
 */
int Tokenize(const char* text, int32_t text_len,
             int32_t* tokens, int32_t n_tokens_max);
```

### 4.3 解码（Decode）

```cpp
/**
 * @brief 将 token ID 序列解码为字符串
 * @param tokens   token ID 数组
 * @param n_tokens token 数量
 * @return 解码后的 UTF-8 文本
 */
std::string Decode(int32_t* tokens, int32_t n_tokens);
```

### 4.4 辅助 API

```cpp
bool GetVocabInfo(VocabInfo* info);         // 词表信息
std::string TokenToPiece(int32_t token);    // Token ID → piece 字符串
bool IsLoaded() const;                      // 检查加载是否成功
```

### 4.5 最小集成示例

```cpp
#include "Tokenizer.h"

int main() {
    Tokenizer tok("/path/to/model_dir");
    if (!tok.IsLoaded()) return 1;

    int32_t ids[256];
    int n = tok.Tokenize("Hello, 世界！", 13, ids, 256);
    // ids[0..n-1] 为 token ID 序列

    std::string text = tok.Decode(ids, n);
    // text == "Hello, 世界！"
    return 0;
}
```

编译：链接 `libtokenizer.a`，加 `-std=c++17`。

---

## 5. 命令行工具（tokenize_demo）

### 5.1 单文本模式

```bash
./tokenize_demo -t /path/to/model_dir -p "人工智能是计算机科学的重要分支。" --show-count
```

输出：
```
vocab_info: vocab_size=151669 n_special_bos_id=1 n_special_eos_id=1
special_bos_id[0] = 151645
special_eos_id[0] = 151645
     104455 -> 'æĤºäººå...'
      20412 -> 'æĶ¯'
      ...
Decode: 人工智能是计算机科学的一个重要分支。
Total number of tokens: 7
```

### 5.2 批量编码模式

```bash
echo -e "2\n5 Hello\n15 你今天好吗" | ./tokenize_demo --stdin-batch -t /path/to/model_dir
```

格式：第一行 = N（总文本数），之后每行 "BYTELEN rawbytes"。输出：每行 "n id1 id2 ... idN"。

### 5.3 批量解码模式

```bash
echo -e "2\n9707 11 1879 0\n108386 3837 99489 6313" | ./tokenize_demo --stdin-decode -t /path/to/model_dir
```

格式：第一行 = N，之后每行空格分隔的 token ID。输出：长度前缀的解码文本行。

---

## 6. 支持的模型

| # | 模型 | HF Model ID | Tokenizer |
|---|------|-------------|-----------|
| 1 | Qwen3 | Qwen/Qwen3-1.7B | BPE |
| 2 | Qwen2.5 | Qwen/Qwen2.5-3B-Instruct | BPE |
| 3 | Qwen2.5-VL | Qwen/Qwen2.5-VL-3B-Instruct | BPE |
| 4 | Qwen2.5-Omni | Qwen/Qwen2.5-Omni-3B | BPE |
| 5 | Qwen3-VL | Qwen/Qwen3-VL-4B-Instruct | BPE |
| 6 | Qwen3-ASR | Qwen/Qwen3-ASR-0.6B | BPE |
| 7 | Qwen3-Embedding | Qwen/Qwen3-Embedding-0.6B | BPE |
| 8 | Qwen3-Reranker | Qwen/Qwen3-Reranker-0.6B | BPE |
| 9 | Qwen3-TTS | Qwen/Qwen3-TTS-12Hz-1.7B-Base | BPE |
| 10 | GME-Qwen2-VL | Alibaba-NLP/GME-Qwen2-VL-2B-Instruct | BPE |
| 11 | HY-MT1.5 | Tencent-Hunyuan/HY-MT1.5-1.8B | BPE |
| 12 | Janus-Pro | deepseek-ai/Janus-Pro-1B | BPE |
| 13 | SmolVLM | HuggingFaceTB/SmolVLM-500M-Instruct | BPE |
| 14 | SmolVLM2 | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | BPE |
| 15 | glm-edge | THUDM/glm-edge-1.5b-chat | BPE |
| 16 | Gemma 4 | google/gemma-4-E2B-it | BPE / SPM |
| 17 | InternVL3 | OpenGVLab/InternVL3-1B | BPE |
| 18 | MiniCPM-V-4 | openbmb/MiniCPM-V-4 | SPM |
| 19 | FastVLM | llava-hf/llava-1.5-7b-hf | SPM |

**模型文件目录要求**：

| 文件 | 用途 | 必需 |
|------|------|------|
| `tokenizer.json` | BPE 词表 + merge rules + pretokenizer 配置 | BPE 模型 |
| `tokenizer_config.json` | bos/eos/unk token 名称 | 推荐 |
| `tokenizer.model` | SentencePiece 模型 | SPM 模型 |

> `save_pretrained()` 会生成以上全部文件，可直接拷贝整个目录使用。

---

## 7. 精度数据

测试条件：40,000 条文本（EN TinyStories 20,000 + ZH Alpaca-ZH 20,000），总计 ~8,270,000 字符。指标为与 HF Python tokenizer 的 token 序列完全匹配率。

### 7.1 三版本对比

| 版本 | 引擎 | 测试平台 | 模型数 | 总体准确率 | 平均 Token F1 | 说明 |
|:-----|:-----|:--------|:------:|:---------:|:-----------:|:-----|
| 原始 model_zoo | llama.cpp GGUF | x86_64 (GCC 13) | 16 | 100.00% | 1.0000 | 需要 GGUF 格式转换再加载 |
| 原始 meta-pytorch | meta-pytorch/tokenizers | x86_64 (GCC 13, C++20) | 19 | 99.70% | 0.9976 | 上游原版，C++20，零格式转换 |
| 改造后 | meta-pytorch/tokenizers (C++17) | RK3588 (GCC 7.4, C++17) | 19 | 100.00% | 1.0000 | 本仓库，C++17 适配，零格式转换 |

### 7.2 分模型精度

| 模型 | 原始 model_zoo (GGUF) | 原始 meta-pytorch (C++20) | 改造后 (C++17, RK3588) |
|:-----|:---:|:---:|:---:|
| Qwen3 | 100.00% | 99.30% | 100.00% |
| Qwen2_5 | 100.00% | 99.30% | 100.00% |
| Qwen2_5_VL | 100.00% | 99.30% | 100.00% |
| Qwen2_5_Omni | 100.00% | 99.30% | 100.00% |
| Qwen3_VL | 100.00% | 99.30% | 100.00% |
| Qwen3_ASR | 100.00% | 99.30% | 100.00% |
| Qwen3_Embedding | 100.00% | 99.30% | 100.00% |
| Qwen3_Reranker | 100.00% | 99.30% | 100.00% |
| Qwen3_TTS | 100.00% | 99.30% | 100.00% |
| GME-Qwen2-VL | 100.00% | 99.30% | 100.00% |
| InternVLM | 100.00% | 99.30% | 100.00% |
| HY_MT_1_5 | 100.00% | 100.00% | 100.00% |
| Janus_Pro | 100.00% | 100.00% | 100.00% |
| SmolVLM | 100.00% | 100.00% | 100.00% |
| SmolVLM2 | 100.00% | 100.00% | 100.00% |
| glm_edge | 100.00% | 100.00% | 100.00% |
| MiniCPM_V_4 | 100.00% | 100.00% | 100.00% |
| FastVLM | 100.00% | 100.00% | 100.00% |
| gemma4 | N/A | 100.00% | 100.00% |

> **说明**：原始 meta-pytorch (C++20) 在 Qwen 系列模型上的 ~99.30% 准确率来自 GCC 13 编译时 NFC normalizer 的 Unicode 组合表构建差异（约 1,400/200,000 条 ZH 文本存在 1-2 字节 token 偏差）。改造后（C++17 版本）修复了 NFC 逻辑，RK3588 上达到 100% 匹配。原始 model_zoo 使用 llama.cpp 的 GGUF 词表格式，完全不同引擎，匹配率 100%，但需要 GGUF 格式转换。

---

## 8. 性能数据

测试条件：固定输入文本长度 ~207 字符（40,000 条 EN+ZH 混合文本平均长度）。速度 = C++ batch encode 单文本耗时。内存 = tokenizer 加载模型后的峰值 RSS。

| 平台 | 编译器 | CPU | 速度 (ms/text) | 峰值 RSS (MB) | 说明 |
|:-----|:------|:----|:-----:|:-----:|:-----|
| x86_64 | GCC 13 | Intel Core i7-14700 | 0.15 | 28.6 | PC 桌面端基准 |
| Linux aarch64 | GCC 7.4 Linaro | RK3588 (4×A76 + 4×A55) | 0.27 | 27.6 | 嵌入式 Linux 板端 |
| Android arm64-v8a | NDK r23c Clang 12 | RK3576 (4×A76 + 4×A55) | 1.18 | 22.1 | 嵌入式 Android 板端 |

### 8.1 分模型性能

| 模型 | x86_64 速度 | RK3588 速度 | RK3576 速度 | x86_64 RSS | RK3588 RSS | RK3576 RSS |
|:-----|-----:|-----:|-----:|-----:|-----:|-----:|
| Qwen3 | 0.15 ms | 0.27 ms | 1.18 ms | 34.5 MB | 29.6 MB | 24.6 MB |
| Qwen2_5 | 0.16 ms | 0.27 ms | 1.21 ms | 34.6 MB | 32.6 MB | 24.7 MB |
| Qwen2_5_VL | 0.16 ms | 0.27 ms | 1.19 ms | 34.4 MB | 34.3 MB | 24.6 MB |
| Qwen2_5_Omni | 0.16 ms | 0.27 ms | 1.18 ms | 34.5 MB | 29.6 MB | 22.1 MB |
| Qwen3_VL | 0.16 ms | 0.27 ms | 1.21 ms | 34.4 MB | 33.8 MB | 24.5 MB |
| Qwen3_ASR | 0.16 ms | 0.27 ms | 1.18 ms | 34.6 MB | 29.4 MB | 24.7 MB |
| Qwen3_Embedding | 0.16 ms | 0.27 ms | 1.18 ms | 34.5 MB | 34.3 MB | 24.6 MB |
| Qwen3_Reranker | 0.16 ms | 0.27 ms | 1.21 ms | 34.5 MB | 33.5 MB | 24.6 MB |
| Qwen3_TTS | 0.16 ms | 0.27 ms | 1.20 ms | 34.4 MB | 33.5 MB | 24.6 MB |
| GME-Qwen2-VL | 0.16 ms | 0.26 ms | 1.24 ms | 34.3 MB | 29.4 MB | 27.2 MB |
| InternVLM | 0.16 ms | 0.27 ms | 1.22 ms | 34.5 MB | 33.8 MB | 27.2 MB |
| HY_MT_1_5 | 0.16 ms | 0.28 ms | 1.26 ms | 29.1 MB | 24.7 MB | 19.6 MB |
| Janus_Pro | 0.21 ms | 0.32 ms | 1.59 ms | 24.6 MB | 20.3 MB | 19.2 MB |
| SmolVLM | 0.12 ms | 0.30 ms | 1.25 ms | 13.6 MB | 6.2 MB | 9.1 MB |
| SmolVLM2 | 0.12 ms | 0.30 ms | 1.48 ms | 13.7 MB | 6.2 MB | 9.2 MB |
| glm_edge | 0.17 ms | 0.26 ms | 1.27 ms | 19.3 MB | 12.9 MB | 13.9 MB |
| MiniCPM_V_4 | 0.06 ms | 0.15 ms | 0.77 ms | 20.1 MB | 13.9 MB | 13.9 MB |
| FastVLM | 0.05 ms | 0.21 ms | 1.04 ms | 10.0 MB | 9.3 MB | 6.4 MB |
| gemma4 | 0.13 ms | 0.22 ms | 1.06 ms | 76.9 MB | 76.6 MB | 56.0 MB |

### 8.2 峰值内存优化

原始的 meta-pytorch/tokenizers 用 `nlohmann/json` 把整个 `tokenizer.json` 解析成内存 DOM 树，加载时峰值内存巨大（Qwen3 约 89 MB、gemma4 约 240 MB）。本仓库通过以下四步把峰值内存压到原来的 1/3 左右：

| # | 优化手段 | 原理 |
|:-:|:---------|:-----|
| 1 | **mmap 映射 + 及时释放** | `tokenizer.json` 用 `mmap(PROT_READ)` 零拷贝映射，不再 `read()` 进 `std::string`。解析完 `munmap` 释放文件页；glibc 端再调 `malloc_trim(0)` 把空闲堆返还系统。 |
| 2 | **手写 JSON 扫描器替代 DOM 树** | `vocab`/`merges` 两个最大 section（十几万条目）用 `find_key_value` / `skip_json_string` / `decode_json_string` 直接流式扫描，不构建 JSON 节点树，消除 nlohmann ~45 MB（或 yyjson ~48 MB）的节点树开销。仅 normalizer/pretokenizer 等小配置段用 nlohmann 解析子区间。 |
| 3 | **MergeMap 扁平化** | merge 规则从 `std::unordered_map<pair,pair,PairHash>` 改为排序 `std::vector<MergeEntry>`，查询用 `std::lower_bound`，消除哈希表节点与桶的开销。 |
| 4 | **索引字段瘦身** | `BuilderElement` 的 `element_offset` / `original_index` 从 `size_t`(8B) 改为 `uint32_t`(4B)，vocab 十几万条目各省 8 字节。 |

**优化前后峰值 RSS 对比**（平均值，40,000 条文本批量编码）：

| 平台 | 优化前 RSS | 优化后 RSS | 降幅 |
|:-----|----------:|----------:|-----:|
| x86_64 | 77.0 MB | 28.6 MB | -48.4 MB（62.9%） |
| Linux aarch64（RK3588） | 80.2 MB | 27.6 MB | -52.6 MB（65.6%） |
| Android（RK3576） | 49.3 MB | 22.1 MB | -27.2 MB（55.2%） |

典型大模型单点对比（RK3588）：

| 模型 | 优化前 RSS | 优化后 RSS | 降幅 |
|:-----|----------:|----------:|-----:|
| Qwen3（151K vocab） | 88.9 MB | 29.6 MB | -59.3 MB（66.7%） |
| gemma4（262K vocab） | 240.5 MB | 76.6 MB | -163.9 MB（68.1%） |
| SmolVLM（49K vocab） | 31.4 MB | 6.2 MB | -25.2 MB（80.3%） |

> 注：gemma4 词汇量最大（约 262K），优化后仍是最占内存的模型，但其 76.6 MB 主要来自不可压缩的词表双向索引（token→ID + ID→token）本身，而非 JSON 解析开销。

---

## 9. 库规格

| 指标 | 值 |
|------|-----|
| 静态库体积 | **2.53 MB**（x86_64, GCC 13） / **5.28 MB**（aarch64, GCC 7.4） |
| C++ 标准 | C++17 |
| 许可证 | BSD-3（上游 meta-pytorch/tokenizers） |
| 运行时依赖 | 无（纯静态链接） |
| 线程安全 | 只读操作线程安全；加载和销毁需外部同步 |

### 9.1 体积组成（x86_64）

| 组件 | 大小 |
|------|------|
| Tokenizers 核心（BPE 引擎 + unicode） | ~1,033 KB |
| RE2 正则引擎 | ~492 KB |
| Protobuf-Lite（SPM 加载） | ~463 KB |
| SentencePiece Processor | ~213 KB |
| Tokenizer PIMPL 封装 | ~10 KB |
| **合计** | **~2.5 MB** |

---

## 10. 测试

### 10.1 本地快速验证

```bash
cd tokenizer
bash build.sh -s linux -a x86 -b Release

pip install tokenizers sentencepiece transformers huggingface_hub

# 基础对齐测试（7 texts × 20 models）
python3 zoo_test2.py

# 深度测试（103 cases × 13 categories）
python3 zoo_test3.py

# 数据集精度 + 时延（EN + ZH 各 20,000）
python3 zoo_test4.py

# 往返保真度测试
python3 zoo_test5.py
```

### 10.2 板端精度 + 延时 + 内存测试

#### Linux aarch64（RK3588）

```bash
# 编译 + 推板
bash build.sh -s linux -a aarch64 -b Release
adb push install/tokenizer_linux_aarch64 /data/local/tmp/
adb push /tmp/rknn3_zoo_test_* /data/local/tmp/models/

# 快速烟雾测试（板端）
adb shell "
DEMO=/data/local/tmp/tokenizer_linux_aarch64/demo/tokenize_demo
for d in /data/local/tmp/models/rknn3_zoo_test_*; do
    timeout 30 \$DEMO -t \$d -p 'hello world' --show-count 2>&1 | grep -q 'Decode:' && echo OK: \$(basename \$d) || echo FAIL: \$(basename \$d)
done
"

# PC 端对比测试（40,000 条数据集，含 F1、延迟、峰值 RSS）
python3 board_test4.py
# → 报告: zoo_test_result/board_test_report_linux_aarch64.md
```

#### Android arm64-v8a（RK3576）

```bash
# 编译 + 推板
bash build.sh -s android -a arm64-v8a -b Release
adb push install/tokenizer_android_arm64-v8a/demo/tokenize_demo /data/local/tmp/
adb push /tmp/rknn3_zoo_test_* /data/local/tmp/models/

# PC 端对比测试
python3 board_test_android.py
# → 报告: zoo_test_result/board_test_report_android.md
```

---

## 11. 常见问题

### Q: 静态库为什么是 2.5 MB？

A: 核心大小来自 BPE 引擎（nlohmann/json 模板展开）、RE2 完整编译和 SentencePiece Protobuf。去掉 SPM 可缩减至 ~1.5 MB。

### Q: 如何只编译 BPE（去掉 SPM）？

A: 修改 `tokenizer/thirdparty/tokenizers/CMakeLists.txt`，注释 SPM 相关的 `add_subdirectory`、源文件、include 路径和链接，删除 `merge_libs.sh` 中 `libsentencepiece.a` 行。

### Q: aarch64 编译体积为什么比 x86_64 大？

A: GCC 7.4 代码生成不如 GCC 13，ARM64 指令编码密度也低于 x86_64，两者叠加约 2x 膨胀。切换到 ARM GCC 13.3 可大幅缩小。

### Q: 为什么用 GCC 7.4 而不是 GCC 13？

A: RK 芯片官方 BSP 使用 Linaro GCC 7.4 工具链。RK3588 板端 GLIBC 2.17，GCC 13 编译的二进制需要 GLIBC 2.38+，不兼容。

### Q: 支持哪些 tokenizer 类型？

A: BPE（ByteLevel）、Metaspace、SentencePiece（SPM）。不支持 Unigram / WordPiece / BPE 非 ByteLevel 变体。

---

## 12. 依赖与许可证

| 组件 | 来源 | 许可证 |
|------|------|--------|
| meta-pytorch/tokenizers | Meta / PyTorch ExecuTorch | BSD-3 |
| RE2 | Google | BSD-3 |
| PCRE2 | Philip Hazel | BSD-3 |
| SentencePiece | Google | Apache 2.0 |
| nlohmann/json | Niels Lohmann | MIT |
| Abseil-Cpp | Google | Apache 2.0（仅编译期 header） |
| llama.cpp-unicode | llama.cpp 项目 | MIT |

所有运行时代码均为 BSD-3 或 MIT 兼容许可证。

# llama_vocab

llama_vocab是llama.cpp（https://github.com/ggml-org/llama.cpp）的 Tokenizer（分词器）实现

## Tokenizer类型

llama_vocab 支持多种类型的 Tokenizer，主要包括：

1. **SPM (SentencePiece Model)** - Llama模型默认使用的tokenizer类型 
2. **BPE (Byte Pair Encoding)** - 用于GPT-2等模型
3. **WPM (WordPiece Model)** - 用于BERT等模型
4. **UGM (Unigram Model)** - 用于T5等模型
5. **RWKV** - 用于RWKV模型

## 编译说明

llama_vocab 工程支持不同在目标平台（Linux、Android、RISC-V）编译 

### 编译脚本说明

build.sh支持的参数选项如下：

| 选项 | 说明                                             | 示例         |
| ---- | ------------------------------------------------ | ------------ |
| `-s` | 目标系统平台（`linux`、`android`、`riscv64`）    | `-s linux`   |
| `-a` | 目标架构                                         | `-a aarch64` |
| `-n` | SDK 名称（默认：`llama_vocab`）                  | `-n my_sdk`  |
| `-b` | 编译类型（`Debug`、`Release`、`RelWithDebInfo`） | `-b Debug`   |
| `-m` | 启用ASAN（选填）                                 | `-m`         |

### 环境设置脚本

根据 `-s` 指定的目标系统，脚本将自动加载以下环境脚本之一：

- Linux 平台：`env_linux.sh`
- Android 平台：`env_android.sh`
- RISC-V 平台：`env_riscv64.sh`

请确保这些脚本存在且正确设置如下变量：

- `C_COMPILER` 和 `CXX_COMPILER`
- 相关工具链路径（例如 `ANDROID_NDK_PATH`）

编译工具版本要求：

* c++：要求C++14及以上版本
* cmake：要求3.14及以上版本

### 编译示例

编译 **Linux / aarch64**：

```
./build.sh -s linux -a aarch64 -b Release
```

编译 **Android / arm64-v8a**：

```
./build.sh -s android -a arm64-v8a -b Release
```

编译 **RISC-V 平台**：

```
./build.sh -s riscv64 -a generic -b Release
```

启用 **ASAN**：

```
./build.sh -s linux -a aarch64 -b Debug -m
```

### 输出目录说明

脚本执行后将：

- 生成构建文件至：
   `./build/build_<sdk>_<system>_<arch>_<build_type>/`
- 安装构建产物至：
   `./install/<sdk>_<system>_<arch>/`

## **词汇表管理**

llama_vocab 使用 GGUF 进行分词器词汇表存储

### 词汇表提取

使用 convert_hf_to_gguf.py 工具可将词汇表从模型中提取出来，示例：

```
python3.9 convert_hf_to_gguf.py --vocab-only ./Llama-2-7b-hf
INFO:hf-to-gguf:Model vocab successfully exported to ./Llama-2-hf-vocab.gguf
```

## 示例程序说明

示例代码的路径为：

```
./demo/tokenize.cpp
```

执行编译后将生成可执行文件 llama-tokenize

llama-tokenize程序的基本使用方法：

```
./llama-tokenize -m <model_path> -p <text>
```

llama-tokenize程序输出示例：

```
./llama-tokenize -m ./Llama-2-hf-vocab.gguf -p "The weather is nice today"

llama_model_loader: loaded meta data with 30 key-value pairs and 0 tensors from ./Llama-2-hf-vocab.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Llama 2 7b Hf
llama_model_loader: - kv   3:                           general.finetune str              = hf
llama_model_loader: - kv   4:                           general.basename str              = Llama-2
llama_model_loader: - kv   5:                         general.size_label str              = 7B
llama_model_loader: - kv   6:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   7:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv   8:                          llama.block_count u32              = 32
llama_model_loader: - kv   9:                       llama.context_length u32              = 4096
llama_model_loader: - kv  10:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv  11:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv  12:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv  13:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv  14:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  15:                          general.file_type u32              = 1
llama_model_loader: - kv  16:                           llama.vocab_size u32              = 32000
llama_model_loader: - kv  17:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - kv  19:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  20:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  21:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  22:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  24:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  26:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  28:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  29:               tokenizer.ggml.add_eos_token bool             = false
init_tokenizer: initializing tokenizer for type 1
load: control token:      2 '</s>' is not marked as EOG
load: control token:      1 '<s>' is not marked as EOG
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 3
load: token to piece cache size = 0.1684 MB
print_info: vocab type       = SPM
print_info: n_vocab          = 32000
print_info: n_merges         = 0
print_info: BOS token        = 1 '<s>'
print_info: EOS token        = 2 '</s>'
print_info: UNK token        = 0 '<unk>'
print_info: PAD token        = 0 '<unk>'
print_info: LF token         = 13 '<0x0A>'
print_info: EOG token        = 2 '</s>'
print_info: max token length = 48
     1 -> '<s>'
   450 -> ' The'
 14826 -> ' weather'
   338 -> ' is'
  7575 -> ' nice'
  9826 -> ' today'
```


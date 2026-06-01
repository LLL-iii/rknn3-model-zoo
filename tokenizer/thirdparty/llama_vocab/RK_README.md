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



## tokenizer 添加新模型说明（以 Qwen3.5 为例）

### 目标与适用范围
本节详细说明了如何在当前项目（基于 GGUF/llama.cpp 架构）中添加对新模型 Tokenizer 的支持。整个流程包含计算特征哈希值、修改 Python 模型转换脚本以及更新 C++ 底层词表解析逻辑。

### 整体改动概览
需要修改的文件主要包括：
1. `llama_vocab/convert_hf_to_gguf.py`
2. `llama_vocab/gguf-py/gguf/constants.py`
3. `llama_vocab/src/llama-arch.cpp`
4. `llama_vocab/src/llama-arch.h`
5. `llama_vocab/src/llama-vocab.cpp`
6. `llama_vocab/src/llama-vocab.h`

整体流程可以概括为：
1. 先通过 tokenizer 的特征 token 序列哈希识别 `qwen35`。
2. 再为 `Qwen3.5` 和 `Qwen3.5 MoE` 注册新的模型架构。
3. 在 GGUF 常量和 llama 内部架构枚举中补齐对应占位。
4. 为 tokenizer 增加新的 pre-type，并接入对应正则。
5. 在 vocab 加载阶段把字符串形式的 tokenizer pre 名称映射到新 pre-type。

### 具体改动说明
1. 在 `llama_vocab/convert_hf_to_gguf.py` 中补充 tokenizer 识别。

在 `get_vocab_base_pre()` 中增加对 `Qwen3.5` tokenizer 的哈希识别逻辑：

```python
if chkhsh == "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4":
   # ref: https://huggingface.co/Qwen/Qwen3.5-9B-Instruct
   res = "qwen35"
```

其中 `chkhsh` 由如下代码生成：

```python
# 这段chktxt来源于convert_hf_to_gguf的get_vocab_base_pre
chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

chktok = tokenizer.encode(chktxt)

chkhsh = sha256(str(chktok).encode()).hexdigest()
```

这一步的作用是将 Hugging Face tokenizer 的预处理行为映射为内部可识别的 `tokenizer_pre` 名称。后续在 C++ 侧会根据这个名称选择对应的 pre-tokenizer 规则。

2. 在同一文件中注册新的模型架构。

增加如下注册：

```python
@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
class Qwen3_5TextModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

```

这里继承 `Qwen2MoeModel` 的关键原因在于当前主要依赖其 `set_vocab()` 等通用逻辑，而不是复用完整模型结构本身。若目标模型与现有模型词表体系一致，这种方式通常足以支持 vocab 导出。如果目标依赖架构缺失，但词表与现有模型兼容，通常可以沿用同源 vocab 方案；由于这里只处理 `vocab-only`，因此可以不补充完整 tensor 定义，只保留必要占位。

3. 在 `llama_vocab/gguf-py/gguf/constants.py` 中补充新架构枚举与名称映射。

在 `MODEL_ARCH` 中增加：

```python
class MODEL_ARCH(IntEnum):
   ...
   QWEN3VL       = auto()
   QWEN3VLMOE    = auto()
   QWEN35        = auto()
   QWEN35MOE     = auto()
   ...
```

在 `MODEL_ARCH_NAMES` 中增加：

```python
MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
   ...
   MODEL_ARCH.QWEN3VL:       "qwen3vl",
   MODEL_ARCH.QWEN3VLMOE:    "qwen3vlmoe",
   MODEL_ARCH.QWEN35:        "qwen35",
   MODEL_ARCH.QWEN35MOE:     "qwen35moe",
   ...
}
```

在 `MODEL_TENSORS` 中增加占位：

```python
MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
   ...
   MODEL_ARCH.QWEN35: [
   ],
   ...
}
```

这里不需要填写具体 tensor 内容，空列表即可。原因是当前目标仅为补齐架构定义与 vocab 路径，不涉及完整 tensor 映射。

4. 在 `llama_vocab/src/llama-arch.h` 与 `llama_vocab/src/llama-arch.cpp` 中增加内部架构定义。

`llama_vocab/src/llama-arch.h` 中，向 `llm_arch` 枚举增加：

```cpp
enum llm_arch {
...
    LLM_ARCH_QWEN35,
...
}
```

`llama_vocab/src/llama-arch.cpp` 中，向 `LLM_ARCH_NAMES` 增加：

```cpp
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
...
    { LLM_ARCH_QWEN35,          "qwen35"},
...
}
```

5. 在 `llama_vocab/src/llama-vocab.h` 中增加新的 tokenizer pre-type。

在 `llama_vocab_pre_type` 中增加：

```cpp
enum llama_vocab_pre_type {
   ...
   LLAMA_VOCAB_PRE_TYPE_QWEN35          = 46,
}
```

这里的枚举值需要与现有定义保持一致，不要与已有 pre-type 冲突。如果你的代码基线中该编号已经被占用，应顺延到新的可用值。

6. 在 `llama_vocab/src/llama-vocab.cpp` 中增加 Qwen3.5 对应的正则规则。

在 `llm_tokenizer_bpe` 结构体的相关分支中增加：

```cpp
case LLAMA_VOCAB_PRE_TYPE_QWEN35:
    regex_exprs = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    break;
```

这部分 regex 可以参考 Hugging Face 模型目录中的 `tokenizer.json`，重点查看`pretokenizers` 或 `pattern` 字段。迁移时要特别注意以下两点：
1. 需要转换成 C++ 侧可用的正则表达式形式。
2. Python / JSON 中的转义字符进入 C++ 字符串后通常需要再次转义。

如果原始 tokenizer 使用的是 `Sequence`，则应确认其中真正影响切分的步骤。以本例来看，核心规则来自 `Split` 的 `Regex`，而 `ByteLevel` 本身也会影响最终 tokenizer 行为，因此只抄正则还不够，仍需在加载阶段正确绑定 pre-type。

7. 在 vocab 加载逻辑中增加 `qwen35` 到 pre-type 的映射。

在 `llama_vocab::impl::load(llama_model_loader & ml, const LLM_KV & kv)` 中，BPE 模型的 pre-tokenizer 处理中增加：

```cpp
void llama_vocab::impl::load(llama_model_loader & ml, const LLM_KV & kv) {
...
   // for now, only BPE models have pre-tokenizers
    if (type == LLAMA_VOCAB_TYPE_BPE) {
        add_space_prefix = false;
        clean_spaces = true;
        ...
        else if (
           tokenizer_pre == "qwen35") {
           pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN35;
           clean_spaces = false;
       }
       ...
...
```

这一步非常关键。前面 Python 侧通过哈希识别得到的 `res = "qwen35"`，最终正是通过这里映射到 `LLAMA_VOCAB_PRE_TYPE_QWEN35`。如果缺少这段逻辑，即使前面的架构注册已经完成，运行时仍无法正确选择对应的分词预处理规则。

### 实现注意事项
1. 优先确认 tokenizer 类型是否为 BPE。只有 BPE 模型才会走这里的 pre-tokenizer 逻辑；若模型使用其他词表类型，需要走不同分支。
2. 正则表达式不要直接照搬 JSON 原文。必须检查转义、字符类以及 C++ 正则兼容性，否则很容易出现编译通过但切分结果错误的问题。
3. `MODEL_TENSORS` 这里只需占位，不要误以为必须补齐 tensor 枚举。当前目标是 vocab 兼容，不是完整模型支持。
4. 除了查看 `tokenizer_config.json`，更应该以 `tokenizer.json` 中的实际 `pre_tokenizer` 定义为准，因为真正的切分规则通常在这里。

### 验证建议
完成修改后，建议至少做以下验证：
1. 使用目标模型重新计算 `chkhsh`，确认能够命中 `qwen35` 分支。
2. 执行 vocab 转换流程，确认不会因为未知架构、未知 pre-type 或缺失映射而报错。
3. 对比 Hugging Face tokenizer 与本地实现对同一段测试文本的编码结果，确保 token 序列一致。
4. 重点验证包含以下内容的文本：
   - 英文缩写和撇号组合，如 `'s`、`'re`、`'ve`
   - 中英文混排
   - emoji 与零宽连接符组合
   - 连续标点
   - 数字及小数形式
   - 换行、空格、制表符

### 结论
为 `Qwen3.5` 添加 tokenizer 支持，本质上是补齐三层映射关系：
1. Hugging Face tokenizer 特征哈希到内部 `tokenizer_pre` 名称的映射。
2. 模型架构名称到 GGUF / llama 内部架构枚举的映射。
3. `tokenizer_pre` 名称到 C++ pre-tokenizer 规则与正则表达式的映射。

只要这三层关系打通，并且 `set_vocab()` 所依赖的同源逻辑可以复用，即使暂时不补完整 tensor 定义，也可以先完成 `vocab-only` 支持。这也是新增同类模型时成本最低、风险最可控的接入方式。
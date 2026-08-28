# RKNN3-Model-Zoo · Tokenizer + Jinja 集成说明

> 版本：V1.0.4 + 新版 Tokenizer + Jinja(ChatTemplate)
> 本文件说明本目录相对官方 RKNN3-Model-Zoo 的**全部集成改动**与**使用方法**。

---

## 一、概述

本目录在官方 RKNN3-Model-Zoo（V1.0.4，f630482）基础上集成了两项关键技术：

| 组件 | 作用 | 替代方案 |
|---|---|---|
| **新版 Tokenizer**（`tokenizer/`） | 从 HuggingFace `tokenizer.json` 目录加载，BPE/SPM 双后端 | 旧 `3rdparty/tokenizer`（llama.cpp/GGUF 后端，需格式转换） |
| **Jinja / ChatTemplate**（`jinja/`） | 板端渲染各模型 `chat_template`，输出与 Python `apply_chat_template` **逐字节一致** | 各示例手写 `system_prompt/prefix/postfix` 拼接 |

### 核心 API

```cpp
// 新版 Tokenizer（HF 模型目录加载）
Tokenizer tok(model_dir);                  // model_dir 下含 tokenizer.json（BPE）或 tokenizer.model（SPM）
tok.IsLoaded(); tok.GetVocabInfo(&info);   // 获取词表/bos/eos
tok.Tokenize(text, len, ids, max);         // 分词
tok.Decode(ids, n);                        // 解码

// ChatTemplate（Jinja 渲染）
ChatTemplate ct(model_dir);                // 优先 chat_template.jinja，否则读 tokenizer_config.json 的 chat_template
ct.IsLoaded();
ct.Render(ctx_json, &prompt);              // ctx_json = {"messages":[...], "add_generation_prompt":true}
```

---

## 二、目录结构变化

```
rknn3-model-zoo/
├── tokenizer/                    【替换】新版 tokenizer 源码（HF 目录加载）
│   ├── include/Tokenizer.h       # 新版 API：Tokenizer(model_dir)
│   ├── src/Tokenizer.cpp
│   ├── compat/                   # C++11 shim
│   └── thirdparty/tokenizers/    # meta-pytorch 裁剪版依赖
├── jinja/                        【新增】ChatTemplate 渲染引擎（Jinja2Cpp 移植）
│   ├── include/ChatTemplate.h
│   ├── src/ChatTemplate.cpp
│   └── thirdparty/{jinja2cpp,json,fmt,nonstd}
├── 3rdparty/
│   ├── tokenizer/                【替换】新版预编译库（头 + 各平台 .a）
│   ├── chat_template/            【新增】ChatTemplate 预编译库
│   └── CMakeLists.txt            【改】注册 LIBCHAT_TEMPLATE/LIBCHAT_TEMPLATE_INCLUDES
├── py_utils/
│   ├── export_llm_helper.py      【改】export_tokenizer 重写（动态加载 export_tokenizer_hf）
│   └── export_tokenizer_hf.py    【新增】统一 tokenizer 导出工具（transformers 下载 + hf-mirror）
├── tools/                        （原 MMBench_EVAL、rknn3_llm_test 保留）
└── examples/<21 个模型>/     
    ├── cpp/                      【改】main.cc + CMakeLists.txt
    └── python/                   【改】export_llm.py
```

---

## 三、集成改动清单

### 1. tokenizer
- 移植新版 tokenizer 源码，替换官方旧版（llama/ggml）。
- **删除**旧依赖：`thirdparty/llama_vocab/`（ggml/gguf）、旧 `src/TokenizerBase.h`、`src/llama/`。
- **必要修复**：`tokenizer/merge_libs.sh` 原用 `strip -s` 删光合并库全局符号，导致外部链接 `Tokenizer::*` 全部 undefined；已改为 `strip --strip-unneeded`。

### 2. jinja / ChatTemplate
- 移植 jinja 引擎（C++11 降级、`JINJA2CPP_CHAT_ONLY` 裁剪、单头文件封装）。

### 3. 3rdparty 库接入
- `3rdparty/tokenizer/include/Tokenizer.h` 覆盖为新版头文件。
- 新增 `3rdparty/chat_template/{include/ChatTemplate.h, lib/<平台>/libchat_template.a}`。
- `3rdparty/CMakeLists.txt` 注册：
  ```cmake
  set(LIBCHAT_TEMPLATE ${CHAT_TEMPLATE_PATH}/lib/${CMAKE_SYSTEM_NAME}/${TARGET_LIB_ARCH}/libchat_template.a PARENT_SCOPE)
  set(LIBCHAT_TEMPLATE_INCLUDES ${CHAT_TEMPLATE_PATH}/include PARENT_SCOPE)
  ```

### 4. export 侧
- 新增 `py_utils/export_tokenizer_hf.py`：
  - 远程模型用 `transformers.AutoTokenizer.from_pretrained` 下载并 `save_pretrained`（走 **hf-mirror.com**，`os.environ["HF_ENDPOINT"]` 在 import 前强制设置，避免 huggingface.co 超时）。
  - 本地模型直接拷贝 `tokenizer.json` 等核心文件。
  - 从 `tokenizer_config.json` 提取 `chat_template` 写出 `chat_template.jinja`。
- `py_utils/export_llm_helper.py::export_tokenizer()` 重写：动态加载同目录 `export_tokenizer_hf.py`；顶部强制 `HF_ENDPOINT` 覆盖整个导出流程。

### 5. 示例适配
- **4 个纯文本 chat**（Qwen2_5、Qwen3、HY_MT_1_5、glm_edge）：`new Tokenizer(model_dir)` + **ChatTemplate 渲染** messages → prompt；`system_prompt/prefix/postfix` 常量保留（仍传 `rknn3_session_set_chat_template` 运行时 API）。
- **12 个多模态**：仅换 `Tokenizer(model_dir)`（prompt 为 `"<image> "+user` 裸格式，image 由 `tensor.image` 占位符机制注入，无手写模板可替换）。
- **5 个非 chat**（Qwen3_ASR、Qwen3_Embedding、Qwen3_Reranker、paddleocr_vl；Qwen3_TTS 无 Tokenizer）：仅换构造。
- 所有 `main.cc` 对 `TOKENIZER_BACKEND_LLAMA` 引用清零；模型 argv 的 `tokenizer_path` 语义改为 **HF 目录**。

### 6. CMakeLists 关键改动
- `${LIBTOKENIZER}` 链接改为 **`-Wl,--whole-archive`**（合并库内 Pcre2Regex 等符号为 local，普通 archive 拉取不到，否则运行时报 "Failed to compile regex with PCRE2"）。
- 4 个 chat 模型额外链接 `${LIBCHAT_TEMPLATE}` + include `${LIBCHAT_TEMPLATE_INCLUDES}`。
- tokenizer 安装改为 `file(GLOB TOKENIZER_DIRS "*.tokenizer/")` + `install(DIRECTORY ...)`（新版导出的是**目录**，旧 `*.tokenizer.gguf` glob 已删除）。

### 7. 其他
- `export_llm.py` 调用处 `.tokenizer.gguf` → `.tokenizer`（输出目录）。

---

## 四、构建方法

### 1. 库编译（在集成目录内）

```bash
# tokenizer：
cd tokenizer
bash build.sh -s linux   -a aarch64
bash build.sh -s linux   -a armhf
bash build.sh -s linux   -a x86
bash build.sh -s android -a arm64-v8a
bash build.sh -s android -a armeabi-v7a

# chat_template：
cd ../jinja
bash build.sh -s linux   -a aarch64
bash build.sh -s linux   -a armhf
bash build.sh -s linux   -a x86
bash build.sh -s android -a arm64-v8a
bash build.sh -s android -a armeabi-v7a
```
> 需要对应的交叉编译工具链。

### 2. 预编译库同步到 3rdparty

```bash
INT=/path/to/integration/rknn3-model-zoo
# tokenizer
for p in linux_aarch64 linux_armhf linux_x86 android_arm64-v8a android_armeabi-v7a; do
  case $p in linux_*) sys=Linux;   arch=${p#linux_};; *) sys=Android; arch=${p#android_};; esac
  cp "$INT/tokenizer/install/tokenizer_$p/lib/libtokenizer.a" "$INT/3rdparty/tokenizer/lib/$sys/$arch/"
done
# chat_template
for p in linux_aarch64 linux_armhf linux_x86 android_arm64-v8a android_armeabi-v7a; do
  case $p in linux_*) sys=Linux; arch=${p#linux_};; *) sys=Android; arch=${p#android_};; esac
  cp "$INT/jinja/install/chat_template_$p/lib/libchat_template.a" "$INT/3rdparty/chat_template/lib/$sys/$arch/"
done
```

### 3. 示例编译（Linux / Android）

```bash
# Linux（RK3588 等，交叉编译需 GCC_COMPILER；x86 用系统编译器）
export GCC_COMPILER=<aarch64-none-linux-gnu 前缀>
bash build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5
bash build-linux.sh -t x86    -a x86_64 -d Qwen2_5   # 快速验证

# Android
export ANDROID_NDK_PATH=<ndk-r23c 路径>
bash build-android.sh -t rk3576 -a arm64-v8a -d Qwen2_5
```

---

## 五、模型导出

```bash
# 方式一：随模型导出（export_llm.py 自动调 export_tokenizer）
cd examples/Qwen2_5/python
python export_llm.py --model_path Qwen/Qwen2.5-3B-Instruct \
  --export_llm_path ../model/llm/Qwen2.5-3B-Instruct.onnx
python export_rknn.py    # onnx → .rknn + .weight

# 方式二：单独导出 tokenizer 目录
python py_utils/export_tokenizer_hf.py Qwen/Qwen2.5-3B-Instruct model/llm/Qwen2.5-3B-Instruct.tokenizer
# 需要使用 modelscope 下载时
python py_utils/export_tokenizer_hf.py --modelscope <model_id> model/llm/<model>.tokenizer
```

**tokenizer 目录约定**（匹配新版加载）：
```
<dir>/
├── tokenizer.json            # 必需（BPE）；存在 tokenizer.model 则走 SPM
├── tokenizer_config.json     # bos/eos + chat_template 字段
├── special_tokens_map.json
└── chat_template.jinja       # ChatTemplate 优先读取
```

> 网络：远程下载默认走 **hf-mirror.com**（`export_tokenizer_hf.py` 与 `export_llm_helper.py` 顶部强制 `HF_ENDPOINT`）；如需其他镜像 `export HF_ENDPOINT=<你的地址>`。modelscope: `export_tokenizer_hf.py`支持`--modelscope`；各 export_llm.py 的 `--modelscope`会自动传递给`export_tokenizer`。

---

## 六、运行方法

```bash
# 纯文本 chat（argc=7）：<model.rknn> <weight> <tokenizer_dir> <embed.bin> <core_mask> <prompt>
./rknn_qwen2_5_demo model/Qwen2.5-3B-Instruct.llm.rknn \
  model/Qwen2.5-3B-Instruct.llm.weight \
  model/Qwen2.5-3B-Instruct.llm.tokenizer \    # 【目录】，不是 .gguf
  model/Qwen2.5-3B-Instruct.llm.embed.bin \
  0xff "你好，介绍一下你自己"

# 多模态（例 MiniCPM_V_4，argc=11）：... <tokenizer_dir> <embed.bin> <vision_core> <llm_core> <image> <prompt>
# 非 chat（例 Qwen3_Embedding，argc=7）：... <tokenizer_dir> <embed.bin> <core_mask> <text>
```

各模型 argv 数量/顺序以 `main.cc` 的 usage 打印为准。

---

## 七、验证状态

| 环节 | 结果 |
|---|---|
| tokenizer 库编译（Cygwin/Linux） | ✅ |
| chat_template 库编译 | ✅ |
| 21 个模型适配（TOKENIZER_BACKEND_LLAMA=0） | ✅ |
| Cygwin 全链路运行（tokenizer 加载 + ChatTemplate 渲染 + 分词） | ✅ |
| RK3588 交叉编译 + install | ✅ |
| 板端运行 | tokenizer正常启动、chat_template渲染OK |

---

## 八、常见问题

| 现象 | 原因 / 解决 |
|---|---|
| 链接报 `undefined reference to Tokenizer::*` | 3rdparty 库是旧版或 merge 用了 `strip -s`；重编并确认 merge_libs.sh 用 `--strip-unneeded` |
| 运行报 `Failed to compile regex with PCRE2` | `${LIBTOKENIZER}` 未用 `--whole-archive` 链接 |
| 编译报 `jump to label 'out' crosses initialization` | ChatTemplate 渲染的 `std::string` 声明在 `goto out` 后；把声明提到首个 `goto out` 前 |
| export 下载超时/403 | 已默认走 hf-mirror.com；或者使用`--modelscope`;仍不行则 `export HF_ENDPOINT=<其他镜像>` 或本地放模型目录 |


---

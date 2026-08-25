# RKNN3 chat_template 渲染引擎（Jinja2Cpp）

基于 [Jinja2Cpp](https://github.com/jinja2cpp/Jinja2Cpp)（Apache-2.0 许可证）的独立 C++ chat_template 渲染引擎，用于 RKNN 板端（RK3588 / RK3576 / armv7）运行时渲染对话提示词。原始工程最低要求 C++14，本仓库已完成 **C++14 → C++11 降级改造**，可兼容更老的工具链（Linaro GCC 6.3 / 7.4、NDK Clang 12、Cygwin GCC 14）编译并部署到嵌入式板端。渲染精度与 Python `transformers.apply_chat_template` **逐字节一致**。

**特性**：
- 从 HF 模型目录直接加载 chat_template（`chat_template.jinja` / `tokenizer_config.json` 的 `chat_template` 字段），零格式转换
- 完整 Jinja2 语义（macro 递归 / namespace / 切片 / 链式下标 / `{% generation %}` 等新式特性），新增模板通常无需改引擎
- 单头文件 API（`ChatTemplate.h`，PIMPL 封装），板端程序只需包含它 + 链接 `libchat_template.a`
- **C++11 兼容**（auto 返回、泛型 lambda、init-capture、s 字面量等全部改写为 C++11 等价形式）
- **JINJA2CPP_CHAT_ONLY 功能裁剪**，去掉 chat_template 用不到的 filter/测试器/语句，静态库 11.37 MB → 9.00 MB
- 合并静态库，所有需要推送板端的库合并为一个 `libchat_template.a`
- 端到端精度 **697/697** 用例（18 模型 × 25 场景）与 Python golden 逐字节一致
- 跨平台：Linux x86_64 / aarch64 / Android arm64-v8a / Windows (Cygwin)

---

## 1. 编译

### 1.1 环境要求

| 工具 | 版本 |
|------|------|
| CMake | ≥ 3.18 |
| GCC / Clang | 支持 C++11（Linux aarch64 用 Linaro GCC 7.4，Android 用 NDK r23c Clang 12） |
| Boost | 1.65+（filesystem/regex/system，thirdparty 自带或 BOOST_ROOT 指定） |
| nlohmann json | 3.11.x（thirdparty 自带，header-only） |
| fmt | 9.x（thirdparty 自带，header-only） |
| nonstd lite | expected/variant/optional/string-view（thirdparty 自带，header-only） |

### 1.2 编译命令

默认构建 **linux_x86**（`bash build.sh` 即等价于 `-s linux -a x86 -b Release`）：

```bash
# Linux x86_64 服务器（默认）
bash build.sh

# Linux aarch64（RK3588，Linaro GCC 7.4 交叉编译）
bash build.sh -s linux -a aarch64 -b Release

# Linux armhf（armv7 板端，Linaro GCC 6.3.1 交叉编译）
bash build.sh -s linux -a armhf -b Release

# Android arm64-v8a（RK3576，NDK r23c Clang 12）
bash build.sh -s android -a arm64-v8a -b Release

# Windows 验证（Cygwin，GCC 14 / C++11）
bash build.sh -s cygwin -a x86 -b Release
```

依赖源码全部放在 `thirdparty/`（jinja2cpp / boost / json / fmt / nonstd），无需联网下载。Boost 会从 `thirdparty/boost` 自动编译（b2，`variant=release -Os -std=c++11`）到 `build/boost_<platform>/` 并缓存；也支持 `BOOST_ROOT` 指向预构建的交叉编译 Boost。

### 1.3 安装目录

编译后在 `install/chat_template_<platform>/` 下生成（**保持整洁**）：

```
install/chat_template_linux_aarch64/
├── demo/
│   └── render_driver        ← 板端测试驱动（静态链接，可独立运行）
├── include/
│   └── ChatTemplate.h       ← 板端程序唯一需要 include 的封装层头
└── lib/
    └── libchat_template.a   ← 合并静态库（jinja2cpp + 封装层 + boost_filesystem 合并为一个）
```

> 测试数据（data/ctx、data/golden、manifest.json）与文档不进 install，板端测试时从 `jinja/data/` 单独推送。

### 1.4 交叉编译 aarch64（RK3588）

需要 ARM GCC 7.4 工具链：

```bash
# env_linux.sh 中确认工具链路径
# RK_AARCH64_TOOLCHAIN=/opt/toolchains/gcc-linaro-7.4.1-2019.02-x86_64-aarch64-linux-gnu/bin/

bash build.sh -s linux -a aarch64 -b Release
file install/chat_template_linux_aarch64/demo/render_driver
# 应输出: ELF 64-bit LSB executable, ARM aarch64
```

### 1.5 交叉编译 armhf（armv7 板端）

需要 Linaro GCC 6.3.1 armhf 工具链：

```bash
# env_linux.sh 中确认工具链路径
# ARMHF_TOOLCHAIN=/opt/toolchains/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/

bash build.sh -s linux -a armhf -b Release
file install/chat_template_linux_armhf/demo/render_driver
# 应输出: ELF 32-bit LSB executable, ARM, EABI5 (hard-float ABI)
```

### 1.6 Android 编译（RK3576）

设置 NDK 路径：

```bash
# env_android.sh
# ANDROID_NDK_PATH=/opt/toolchains/android-ndk-r23c

bash build.sh -s android -a arm64-v8a -b Release
file install/chat_template_android_arm64-v8a/demo/render_driver
```

---

## 2. C++14 → C++11 降级说明

原始 Jinja2Cpp 最低要求 C++14（CMake `FATAL_ERROR ... LESS 14`）。为兼容更老的工具链（GCC 7.4 / NDK / 嵌入式板端），本仓库做了 C++11 降级改造（纯改代码，不引入兼容头遮蔽）：

| 原始（C++14） | 改造后（C++11） | 涉及 |
|:---|:---|:---|
| 函数返回 `auto`（`auto f() {...}`） | 尾置 `-> decltype(...)` 或具体返回类型 | 全部 .cpp/.h（~40 处） |
| 泛型 lambda `[](auto x)` | 命名空间作用域模板仿函数（C++11 禁止局部类模板成员）或具体类型 | filters / testers / statements / string_converter_filter / binding（~25 处） |
| init-capture `[x = expr]` | 外移为局部变量后按值捕获 `[x]` | internal_value / reflected_value / template_impl / statements（~40 处） |
| `"..."s` 字面量 | `std::string("...")`（`string_literals_compat.h` 已空化为 no-op） | 多处 |
| `std::decay_t / enable_if_t / conditional_t` | `typename std::decay/enable_if/conditional<>::type` | 全部 |
| `std::shared_timed_mutex` / `shared_lock` | `std::mutex` / `unique_lock` | template_env.h |
| `std::is_final` | `__is_final` | polymorphic_cxx14.h |
| constexpr 成员 `operator=`（隐式 const） | 去掉 constexpr | polymorphic_cxx14.h |
| 完美转发构造函数 `X(U&&)` 劫持拷贝/移动 | `enable_if` 排除自身类型 | internal_value.cpp 的 5 个 Adapter 类 |

**关键要点**：
- **C++11 禁止局部类成员模板**（编译期报"局部类中对成员模板的声明无效"），因此泛型 lambda 无法改成局部仿函数，必须提取为命名空间作用域的模板仿函数（`CaseConverter` / `SubstringTest` / `IncludedRendererFactory` 等）。
- 泛型 lambda 的参数类型在多数调用点固定（如 `const InternalValue&`、`const Item&`、`const ValuesMap&`），可直接具体化；真正参数化的（字符串转换回调、模板访问回调）才用仿函数。
- Adapter 类的 `template<typename U> XxxAdapter(U&&)` 完美转发构造会在 C++11 下劫持拷贝/移动构造，导致按值捕获 lambda 编译失败；用 `enable_if` 排除 `decay<U> == 自身类型` 修复。
- **降级过程中发现并修复了 56 处历史 s-literal 替换遗留的字符串损坏**（`"mapstd::string("`、`")tring"`、`")uper"`、`")et"` 等），涉及 filter/tester/关键字注册表 key、`self`/`super`/`set`/`strip`/`step` 等。这些损坏即使编译通过也会导致功能错误，已按白名单精确修复。

---

## 3. 体积优化

### 3.1 静态库体积（三平台实测）

| 阶段 / 平台 | 体积（含符号） |
|:---|:---:|
| 原始完整 Jinja2Cpp 合并库（裁剪前） | ~11.37 MB |
| **JINJA2CPP_CHAT_ONLY 裁剪**（只保留 chat_template 用到的 filter/test/语句） | ~9.0 MB（Cygwin） |
| **优化前** linux_x86 / aarch64 / android | 8.66 / 9.55 / 11.23 MB |
| **优化后** linux_x86（GCC 13） | **~7.99 MB**（减 0.67） |
| **优化后** linux_aarch64（GCC 7.4） | **~8.86 MB**（减 0.69） |
| **优化后** android_arm64-v8a（Clang 12） | **~10.48 MB**（减 0.75） |


- 裁剪手段：`JINJA2CPP_CHAT_ONLY` 宏排除非 chat_template 所需的 filter（title/truncate/escape/wordcount 等）、测试器（defined/iterable 等）与语句，对应代码不参与编译。
- `-Os -DNDEBUG`；Boost `variant=release`（避免 Android 曾 96 MB 的 debug 膨胀）；merge 时 `strip --strip-unneeded`。
- **进一步优化（已实施，共减 ~0.7MB/平台）**：
  - **P1 裁剪 wchar_t 实例化**：`TemplateW`（宽字符模板）实现、`LoadTemplateW`、相关 `IsEqual` 用 `#ifndef JINJA2CPP_CHAT_ONLY` 包裹，`TemplateImpl<wchar_t>` 不再实例化（template 组件 1.67→1.13MB）。chat 只用 char。
  - **错误处理 Release 裁剪**：`ValueRenderer`/fmt formatter 仅 Debug（`NDEBUG`）编译；Release 错误信息只输出错误码（error_info 组件 0.39→0.32MB）。
- 体积大头是完整引擎代码 + 链接必需的符号/重定位（解包 .o 共 ~6.9MB，.a 归档再加 ~1.1MB 索引）。aarch64/android 比 x86 大因旧工具链代码密度低 / libc++ 模板膨胀。**已接近体积下限，进一步减需深度功能裁剪，维护成本高**（详见 §9.1）。

---

## 4. API

### 4.1 构造函数

```cpp
/**
 * @brief 从 HF 模型目录加载模板
 * @param model_dir 模型目录（含 chat_template.jinja 或 tokenizer_config.json）
 *
 * 加载优先级（与 transformers 一致）：
 *   <model_dir>/chat_template.jinja（若存在）→
 *   <model_dir>/tokenizer_config.json 的 chat_template 字段
 *   （支持字符串形式，或 {"default": "..."} 对象形式）。
 */
explicit ChatTemplate(const char* model_dir);

/**
 * @brief 从模板源码字符串加载（UTF-8）
 * @param tpl     模板源码
 * @param tpl_len 字节数
 *
 * 末尾换行按 keep_trailing_newline=False 自动处理（对齐 Python）。
 */
ChatTemplate(const char* tpl, size_t tpl_len);
```

### 4.2 渲染

```cpp
/**
 * @brief 渲染对话上下文，输出渲染后的 prompt
 * @param ctx_json  对话上下文 JSON（同 transformers.apply_chat_template 输入）：
 *                  {"messages":[...], "add_generation_prompt":true,
 *                   "tools":[...], "bos_token":"...", ...}
 * @param ctx_len   JSON 字节数
 * @param out       输出渲染后的 prompt
 * @return true 渲染成功；JSON 解析或模板渲染失败返回 false
 */
bool Render(const char* ctx_json, size_t ctx_len, std::string* out);

// 便捷重载
bool Render(const std::string& ctx_json, std::string* out);

bool IsLoaded() const;   // 检查模板是否成功加载
```

### 4.3 最小集成示例

```cpp
#include "ChatTemplate.h"

int main() {
    // 从模型目录实时加载模板（优先 chat_template.jinja，否则 tokenizer_config.json）
    ChatTemplate ct("/path/to/model_dir");
    if (!ct.IsLoaded()) return 1;

    const char* ctx = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                      "\"add_generation_prompt\":true}";
    std::string prompt;
    if (!ct.Render(ctx, &prompt)) return 1;
    // prompt 即为渲染后的输入 prompt
    return 0;
}
```

编译：`g++ -std=c++11 -I<install>/include app.cpp <install>/lib/libchat_template.a -lpthread`

---

## 5. 命令行工具（render_driver）

### 5.1 参数

```
render_driver --model-dir <model_dir> --ctx <ctx.json> [--bench N] [--mem] [--log-level <level>]
render_driver --tpl <chat_template.jinja | tokenizer_config.json> --ctx <ctx.json>
render_driver --template "<inline template>" --ctx <ctx.json>
```

| 参数 | 说明 |
|------|------|
| `--model-dir <dir>` | 模型目录，内部按优先级加载模板（模拟板端实时加载） |
| `--tpl <path>` | 模板文件（.jinja 内容即模板；JSON 取 chat_template 字段） |
| `--template <str>` | 内联模板源码 |
| `--ctx <path>` | 对话上下文 JSON 文件；缺省按空上下文渲染纯模板 |
| `--bench N` | 渲染 N 次求平均耗时（stderr 输出 `[bench] ... avg X us`） |
| `--mem` | 打印进程峰值 RSS（`[mem] ru_maxrss=...`） |
| `--log-level` | `off/error/warn/info/debug` 运行期覆盖（Release 默认 Warn，Debug 默认 Debug） |

渲染结果写 **stdout**（不污染），日志/统计写 **stderr**。

### 5.2 单命令行模式

从模型目录实时加载模板并渲染单个 ctx（板端/PC 同命令；默认 `bash build.sh` 构建的 linux_x86 即本模式）：

```bash
./install/chat_template_linux_x86/demo/render_driver \
    --model-dir models/rknn3_zoo_test_gemma4 \
    --ctx data/ctx/rknn3_zoo_test_gemma4_multi_v1.json
```

stdout 只输出渲染结果（日志在 stderr，不污染）：

```
<bos><|turn>system
You are a helpful assistant.<turn|>
<|turn>user
What is artificial intelligence?<turn|>
<|turn>model
I'm doing great! AI is the simulation of human intelligence in machines. Letme elaborate.<turn|>
<|turn>user
Tell me more about it.<turn|>
<|turn>model
```

### 5.3 Release 与 Debug 模式对比

默认日志级别由构建类型决定（写 stderr，不污染 stdout 的渲染结果）：

| 构建类型 | 默认日志级别 | 行为 |
|---------|-------------|------|
| **Release**（`-b Release`，NDEBUG） | Warn | 只报错误/警告，stdout 干净 |
| **Debug**（`-b Debug`） | Debug | 打印详细日志（模板来源/大小、ctx 大小、渲染字节数等） |

加 `--log-level off|error|warn|info|debug` 可运行期覆盖。

Debug 构建（或 `--log-level debug`）stderr 输出示例：

```
[CT][I] main:88 log level = 4
[CT][I] ChatTemplate:75 load template from models/rknn3_zoo_test_gemma4/chat_template.jinja (18569 bytes)
[CT][I] ChatTemplate:94 template loaded
[CT][I] main:110 ctx: 400 bytes from data/ctx/rknn3_zoo_test_gemma4_multi_v1.json
[CT][D] Render:134 Render: ctx parsed, 4 keys
[CT][D] Render:146 Render: 273 bytes output
[CT][D] main:118 render ok, 273 bytes output
```

### 5.4 其他用法

```bash
# 渲染 + 计时 + 内存
./render_driver --model-dir <model_dir> --ctx <ctx> --bench 1000 --mem

# --tpl 直接指定模板文件（.jinja 或 tokenizer_config.json）/ --template 内联模板
./render_driver --tpl chat_template.jinja --ctx <ctx>
./render_driver --template "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}" --ctx <ctx>
```

---

## 6. 支持的模型

模型来自 `<repo>/jinja/data/models/rknn3_zoo_test_*/`（21 个目录，其中 18 个含 chat_template，3 个无 chat_template 自动跳过）：

| # | 模型 | 模板来源 |
|---|------|---------|
| 1 | Qwen3 | tokenizer_config.json |
| 2 | Qwen3-5 | tokenizer_config.json |
| 3 | Qwen3-VL | tokenizer_config.json |
| 4 | Qwen3-VL-LoRA | tokenizer_config.json |
| 5 | Qwen3-Embedding | tokenizer_config.json |
| 6 | Qwen3-Reranker | tokenizer_config.json |
| 7 | Qwen2.5 | tokenizer_config.json |
| 8 | Qwen2.5-VL | tokenizer_config.json |
| 9 | Qwen2.5-Omni | tokenizer_config.json |
| 10 | GME-Qwen2-VL | tokenizer_config.json |
| 11 | HY-MT1.5 | tokenizer_config.json |
| 12 | InternVLM | tokenizer_config.json |
| 13 | MiniCPM-V-4 | tokenizer_config.json |
| 14 | SmolVLM | tokenizer_config.json |
| 15 | SmolVLM2 | tokenizer_config.json |
| 16 | glm-edge | tokenizer_config.json |
| 17 | gemma4 | chat_template.jinja |
| 18 | FastVLM | tokenizer_config.json（含 `{% generation %}` 标签） |

> 跳过（无 chat_template）：Janus-Pro、Qwen3-ASR、Qwen3-TTS。

**模板文件目录要求**：

| 文件 | 用途 | 必需 |
|------|------|------|
| `chat_template.jinja` | 模板源码（若存在则优先） | 推荐 |
| `tokenizer_config.json` | `chat_template` 字段（字符串或 `{"default":...}`）+ bos/eos 注入 | 有模板时必需 |

> render_driver 从模型目录**实时加载模板**，与真实部署一致：板端只放模型目录的配置文件，不需要额外模板旁路。

---

## 7. 精度数据

测试方式：`gen_test_cases.py` 用 Python jinja2 生成 golden（对齐 transformers 环境：`trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False`），`board_test_chat.py` 逐字节对比 C++ 渲染输出。

### 7.1 总体结果

| 项目 | 数值 |
|:-----|:-----|
| 用例数 | **697/697 PASS**（0 FAIL） |
| 覆盖 | **18 模型 × 25 场景**（每场景 1~4 变体，不适配变体自动 SKIP） |
| 判定标准 | C++ 输出与 Python golden **逐字节一致**（含空白/换行/Unicode） |

### 7.2 各平台精度对比

| 平台 | 精度 |
|:-----|:---:|
| Windows x86_64 (Cygwin) | **697/697 PASS** |
| Linux x86_64 | **697/697 PASS** |
| Linux aarch64（RK3588） | **697/697 PASS** |
| Android arm64-v8a（RK3576） | **697/697 PASS** |

### 7.3 场景矩阵

| 类别 | 场景 |
|------|------|
| 基础 | single / multi / nogen / long / empty / longtext(200k) / thinking(enable_thinking true+false) |
| function-calling | tools（单次调用闭环）/ multi_tool（连续多次调用）/ empty_tools（tools=[]）/ tools_no_call（有工具未调用） |
| 边界与健壮性 | system_dynamic（动态/缺失/空）/ system_only / assistant_end_missing / role_alternation（非标准顺序）/ multi_system（多条 system）/ empty_content（content 为 空串/空 list） |
| 内容 | special_chars（换行/引号/反斜杠/Unicode/HTML）/ injection（用户输入含 `{{` `{%`）/ template_vars（额外变量）/ multimodal（图像占位符）/ multimodal_image_url（image 带 URL/base64） |
| 逻辑分支 | advanced_logic（for/if 分支边界：仅末尾/空角色/推理内容） |
| 性能 | long_hist（30 轮）/ long_hist_long_input（长历史+50k/100k 输入） |

### 7.4 关键特性覆盖（Jinja2Cpp 补丁后）

`macro`（含递归）、`namespace` 赋值、`set ns.attr =`、`not in`、`is not none`、`None` 常量、链式下标/调用 `f(x)[0].y`、切片 `list[1:]`、相邻字符串字面量拼接、`map.get(key)`、单引号字符串、`{% generation %}` 标签、字符串方法（`.upper/.lower/.split/.strip/.lstrip/.rstrip/.startswith/.endswith`）、`trim_blocks/lstrip_blocks`、`tojson`/`dictsort`/`selectattr`/`map`/`default` 等过滤器。

---

## 8. 性能数据

测试条件：697 用例（18 模型 × 25 场景）逐字节对比，与 Python golden 一致。延迟 = render_driver `--bench` 平均耗时。内存测量：Cygwin 用 `--mem` 自报 `ru_maxrss`；Linux/Android 用板端脚本 `/proc VmRSS` 采样峰值（linux_x86 为本机模拟但用板端采样方式，故数值偏高）。

| 平台 | 编译器 / 工具链 | 执行方式 | 精度 | 渲染耗时(ms) min/max/avg | 峰值内存(MB) min/max/avg |
|:-----|:-----|:-----|:---:|:---:|:---:|
| Cygwin x86_64（Windows 本机） | GCC 14，C++11 | 本机模拟 | 697/697 | 0.007 / 7.674 / **0.129** | 5.82 / 8.77 / **6.16** |
| Linux x86_64 | GCC 13，C++11 | 本机模拟 | 697/697 | 0.004 / 5.005 / **0.137** | 13.63 / 14.24 / **14.12** |
| Linux aarch64（RK3588） | Linaro GCC 7.4，C++11 | adb 板端 | 697/697 | 0.009 / 11.012 / **0.338** | 3.30 / 5.43 / **3.68** |
| Android arm64-v8a（RK3576） | NDK r23c Clang 12，C++11 | adb 板端 | 697/697 | 0.021 / 42.879 / **1.104** | 4.04 / 6.52 / **4.35** |

> 四平台均 **697/697 与 Python golden 逐字节一致**。
> 最慢场景为 `longtext`（200k 长文本）与最复杂模板 gemma4（linux_aarch64 上 gemma4 `long_hist` 单点约 5ms、longtext 约 47ms）。普通场景 aarch64 上 avg 约 0.2ms。
> 多模态模板（SmolVLM/SmolVLM2 把 `message.content` 当 list 遍历）已自动用 `[{"type":"text",...}]` 格式生成 ctx，避免逐字符遍历导致的渲染退化。
> 目标：渲染 <1ms、内存 <20MB —— 板端 aarch64/android 渲染 avg ~1ms（含进程冷启动，实际渲染在 1ms 内），内存远低于目标。

---

## 9. 库规格

| 指标 | 值 |
|------|-----|
| 静态库体积 | linux_x86 **~7.99 MB** / linux_aarch64 **~8.86 MB** / android_arm64-v8a **~10.48 MB**（含符号，已优化） |
| C++ 标准 | **C++11**（原始 Jinja2Cpp 要求 C++14，本仓库已降级） |
| 许可证 | Apache-2.0（Jinja2Cpp）+ BSL-1.0/MIT（依赖） |
| 运行时依赖 | 无（静态链接，仅合并 jinja2cpp + 封装层） |
| 封装层 | 单头文件 `ChatTemplate.h`（PIMPL），板端只需它 + 库 |

### 9.1 体积构成与优化空间

| 项 | 大小 | 说明 |
|------|------|------|
| 机器码（`.text` 等） | ~7.0 MB（纯内容下限） | 完整 Jinja2Cpp 引擎（lexer/parser/evaluator/filters/testers） |
| 符号表 + 重定位 + 归档元数据 | ~1.0 MB | 链接所需，`strip --strip-unneeded` 已尽量压缩 |
| 三平台差异 | aarch64 比 x86 大 ~0.9MB、android 比 x86 大 ~2.5MB | 旧工具链代码密度低 / libc++ 模板膨胀 |

> **体积说明**：静态库体积大头是完整 Jinja2Cpp 引擎的代码 + 链接必需的符号/重定位信息（解包后 .o 共 ~6.9MB，.a 归档再占 ~1.1MB 索引）。已做的裁剪：`JINJA2CPP_CHAT_ONLY`（去掉非 chat_template 的 filter/测试器/语句）、`-Os -DNDEBUG`、**wchar_t 实例化裁剪 + 错误处理 Release 裁剪**、merge 时 `strip --strip-unneeded`。体积已接近下限（~8MB），进一步减需深度功能裁剪或换更轻引擎，维护成本高，不建议。最终板端链接时用 `-Wl,--gc-sections` 可裁掉未用代码（render_driver 已启用，strip 后 ~1.9MB）。

---

## 10. 测试

### 10.1 本地快速验证（PC 模拟板端）

```bash
cd jinja

# 1) 生成板端测试数据（多场景 ctx + Python golden + manifest）
python gen_test_cases.py
python gen_test_cases.py --list-scenarios    # 查看 19 个场景及变体数

# 2) 本机模拟（install 传路径；Linux 下直接执行，Windows 需设置 CYGWIN_BIN 才走 cygwin 包装）
python board_test_chat.py --install < install 路径 > --local --bench 5000
# 例：python board_test_chat.py --install install/chat_template_linux_x86/ --local --bench 5000
# → PASS 697/697 即与 Python golden 逐字节一致

# 只测某模型
python board_test_chat.py --install < install 路径 > --local --only gemma4,Qwen3
```

### 10.2 板端测试（RK3588 / RK3576）

```bash
# 交叉编译 + 推送
cd jinja
bash build.sh -s linux -a aarch64 -b Release      # 或 android
adb shell mkdir -p /data/local/tmp/chat_template_test/
# 测试脚本 board_test_chat.py 会自动推送测试所需文件

# 板端测试（adb 模式，逐模型实时加载渲染 + 对比 + 数据）
python board_test_chat.py --install <install 路径>
# 例：python board_test_chat.py --install install/chat_template_linux_aarch64/
# → 报告: data/board_test_report.md
```

状态判定：`PASS`（逐字节一致）/ `DIFF`（渲染成功不一致，附字节数）/ `RENDER-ERR` / `GOLDEN-MISS`。

---

## 11. 常见问题

### Q: 静态库为什么是 9~12 MB？

A: 体积大头是**完整 Jinja2Cpp 引擎的代码**（lexer/parser/evaluator/filters，纯内容 ~6.9 MB）+ **链接必需的符号表/重定位/归档索引**（~1.1 MB）。解包后 .o 共 ~6.9 MB，.a 归档再加 ~1.1 MB 索引。`JINJA2CPP_CHAT_ONLY` 裁剪、`-Os`、merge 时 `strip --strip-unneeded` 均已启用。若要 <5 MB 需深度功能裁剪或换更轻引擎（维护成本高）。

### Q: 如何进一步裁剪体积？

A: 已在做的是 `JINJA2CPP_CHAT_ONLY` 宏（排除非 chat_template 的 filter/测试器/语句）。还可：不合并 boost_filesystem（约 -0.5MB，渲染路径不触发文件系统 API）；或裁剪 Jinja2Cpp 源码（约 -1~2MB，升级需重打补丁）。

### Q: 为什么降级到 C++11？

A: 为兼容更老工具链（Linaro GCC 6.3 / 7.4、NDK r23c / 嵌入式板端）。Jinja2Cpp 原始要求 C++14，本仓库已完成 C++11 改造（详见 §2），697/697 测试验证功能无损。

### Q: 板端渲染结果与 golden 不一致？

A: 用 `--local` 本地重现；检查模型目录配置/ctx 推送是否完整；单用例手动复现 + `--log-level debug` 定位（`--model-dir` + `--ctx` 直接跑）。

### Q: 自己程序链接 libchat_template.a 报缺符号？

A: 需 `-lpthread`；用封装层只需 `-I<install>/include`（无需手动处理 nlohmann/Reflect）。

### Q: 升级 Jinja2Cpp 需要重打补丁吗？

A: 需要。RKNN3 适配补丁分四类（功能 / C++11 降级 / 构建裁剪 / 三平台修复，详见 §12），升级时需全部重打。

---

## 12. Jinja2Cpp 补丁清单（相对上游 v1.3.2）

> 升级 Jinja2Cpp 需重打以下全部补丁。功能补丁集中于 expression_parser / expression_evaluator / internal_value / template_parser / statements 五个文件；C++11 降级与三平台修复分散在全部源文件。改动基线：上游 [jinja2cpp/Jinja2Cpp](https://github.com/jinja2cpp/Jinja2Cpp) v1.3.2（Apache-2.0）。

### 12.1 功能补丁（RKNN3 chat_template 适配）

| # | 文件 | 补丁 | 原因 |
|---|------|------|------|
| 1 | `thirdparty/external_boost_deps.cmake` | fallback FindBoost 去掉 `json` 组件 + 手动创建 Boost 组件 targets | Cygwin Boost 1.66 无 boost::json；CMake 4.x FindBoost 对 header-only 组件误判 |
| 2 | `src/serialize_filters.cpp` | `fmt::basic_format_arg<C>(t)` → `fmt::detail::make_arg<C>(t)` | fmt 7+ 移除 basic_format_arg 公开构造 |
| 3 | `src/binding/*_json_parser.h`、`src/template_impl.h` | `boost::anys::unique_any` → `boost::any` | Boost 1.66 无 unique_any（1.70+ 才有） |
| 4 | `src/expression_parser.cpp` | `ParseLogicalCompare` 支持 `not in` | 模板大量用 `key not in list` |
| 5 | 同上 | `ParseSet` 支持 `set ns.attr = expr`（namespace 成员赋值） | gemma4 用 12 处 namespace 状态维护 |
| 6 | `src/statements.h/.cpp` | `SetLineStatement` 支持成员赋值（SetValue 到 namespace map） | 同上 |
| 7 | `src/expression_evaluator.h/.cpp` | 新增 `NamespaceFn` + `CallGlobalNamespace`（namespace() 返回可变 map） | Jinja2Cpp 无 namespace 内建 |
| 8 | 同上 | `IsExpression` 支持 negate（`is not <test>`） | 模板用 `is not none` 等 |
| 9 | `src/expression_parser.cpp` | `is` 后允许 `none/true/false` 关键字作为测试名 | `is none` 原版不支持 |
| 10 | 同上 | `ParseValueExpression` 后缀改为 while 循环（`f(x)[0].y` 链式） | 模板用 `part.split(...)[0]` |
| 11 | 同上 | `Token::None` 解析为 EmptyValue 常量 | `namespace(foo=None)` |
| 12 | `src/internal_value.h/.cpp`、`expression_evaluator.h`、`expression_parser.cpp`、`value_visitors.h` | 新增 `Slice` 类型 + `SliceExpression` + `list[start:stop:step]` 切片 | 模板用 `messages[1:]` |
| 13 | `src/expression_parser.cpp` | 相邻字符串字面量拼接（`"a" "b"` → `"ab"`） | gemma4 跨行字符串 |
| 14 | `src/internal_value.cpp` | `MapAdapter.get(key)` 方法 | 模板大量用 `message.get('content')` |
| 15 | `src/template_parser.h` | `RM_ExprEnd/RM_StmtEnd` 用 `IsInsideQuote` 替代 `=='\''` hack | 修复 `{{'x'}}`（单引号字符串紧跟 }} 被误判） |
| 16 | 同上 | `{% generation %}`/`{% endgeneration %}` 标签 | FastVLM 等 HF 新式标签 |
| 17 | `include/jinja2cpp/template_env.h` | `trimBlocks/lstripBlocks` 默认改 `true` | 对齐 Python/transformers 环境 |
| 18 | `src/internal_value.cpp` | 字符串方法 `.upper/.lower/.split/.strip/.lstrip/.rstrip/.startswith/.endswith` | Jinja2Cpp 原不支持字符串方法调用 |


### 12.2 C++11 降级改造（纯改代码）

| 类别 | 方案 | 涉及 |
|------|------|------|
| auto 返回函数（~40 处） | 尾置 `-> decltype(...)` 或具体类型 | 全部 .cpp/.h |
| 泛型 lambda（~25 处） | 命名空间模板仿函数（C++11 禁局部类模板成员）或具体类型 | filters / testers / statements / string_converter_filter / binding |
| init-capture `[x = ...]`（~50 处） | 外移局部变量按值捕获 | internal_value / reflected_value / template_impl / statements / binding / generic_list_impl |
| `"..."s` 字面量 | `std::string(...)` | 全部 |
| `std::decay_t/enable_if_t/conditional_t` | `typename std::decay/enable_if/conditional<>::type` | 全部 |
| `std::shared_timed_mutex` / `shared_lock` | `std::mutex` + `unique_lock` | template_env.h/.cpp |
| `std::is_final` / constexpr 成员 `operator=` | `__is_final` / 去 constexpr | polymorphic_cxx14.h |
| 完美转发构造劫持拷贝/移动（5 个 Adapter 类） | `enable_if` 排除自身类型 | internal_value.cpp |

> 详见 §2 降级说明。已修复 56 处历史 s-literal 替换遗留的字符串损坏。

### 12.3 构建与功能裁剪

| 文件 | 改动 |
|------|------|
| `CMakeLists.txt` | C++14→C++11（标准 + FATAL 检查降为 11）；`JINJA2CPP_CHAT_ONLY=1` 宏；`filesystem_handler.cpp` 裁剪（死代码）；`-include string_literals_compat.h` |
| `src/string_literals_compat.h`（新增） | C++11 `operator""s` 兜底（现已空化为 no-op，代码中 `"..."s` 均已替换为 `std::string(...)`） |
| 顶层 `CMakeLists.txt` | `-Os`、合并 `libchat_template.a`、Cygwin render_driver 静态链接 `libstdc++/libgcc` |
| `src/template.cpp`、`src/template_env.cpp`、`include/jinja2cpp/template_env.h` | **体积优化：wchar_t 裁剪**——`TemplateW`（宽字符模板）实现、`LoadTemplateW`、`IsEqual` 的 wchar 比较用 `#ifndef JINJA2CPP_CHAT_ONLY` 包裹，`TemplateImpl<wchar_t>` 不再实例化（chat 只用 char；template 组件 1.67→1.13MB） |
| `src/error_info.cpp` | **体积优化：错误处理 Release 裁剪**——`ValueRenderer`/fmt formatter 仅 `NDEBUG` 编译；Release 错误信息只输出错误码（error_info 组件 0.39→0.32MB） |

### 12.4 三平台交叉编译修复

| 平台 | 文件 | 修复 |
|------|------|------|
| Linux x86（链接缺符号） | `CMakeLists.txt`、`src/template_env.cpp`、`include/jinja2cpp/template_env.h` | 恢复 `template_env.cpp` 编译（被裁剪导致 `TemplateEnv::LoadTemplate` 链接缺失）；template_env.cpp C++11 降级；`LoadTemplateImpl` 声明返回类型修正为 `TemplateEnvResult<CharT>` |
| Linux aarch64（GCC 7.4） | `src/error_info.cpp` | `format` 尾置 decltype 去掉字符串字面量（GCC 7 不支持模板签名含字面量） |
| Android（Clang 12） | `src/string_literals_compat.h`、`src/error_info.cpp` | `operator""s` 空化（Clang 将非下划线后缀当 error）；`constexpr void` 去 constexpr（C++11 不允许） |

---

## 13. 依赖与许可证

| 组件 | 来源 | 许可证 |
|------|------|--------|
| Jinja2Cpp | jinja2cpp/Jinja2Cpp | Apache-2.0 |
| Boost | boost.org | BSL-1.0 |
| fmt | fmtlib/fmt | MIT |
| nlohmann/json | nlohmann/json | MIT |
| nonstd lite（expected/variant/optional/string-view） | martinmoene | BSL-1.0 |
| robin_hood | martinus/robin-hood-hashing | MIT |

所有运行时代码均为 Apache-2.0 / BSL-1.0 / MIT 兼容许可证。

---

## 14. 相关文件

| 文件 | 说明 |
|------|------|
| `build.sh` | 统一构建（`-s linux/android/cygwin -a <arch>`，产物整合到 install/） |
| `CMakeLists.txt` | 顶层构建（jinja2cpp + 封装层 + 合并库 + render_driver） |
| `include/ChatTemplate.h` | **RKNN3 封装层头**（板端程序唯一需要 include） |
| `demo/render_driver.cpp` | 渲染驱动（通过封装层接口） |
| `gen_test_cases.py` | 测试数据生成（多场景 ctx + Python golden + manifest） |
| `board_test_chat.py` | 板端测试脚本（adb / --local，Linux 与 Android 通用） |
| `data/` | 测试数据（manifest.json / ctx/ / golden/ / 报告） |


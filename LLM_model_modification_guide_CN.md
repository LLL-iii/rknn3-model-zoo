# LLM 模型适配说明

对于 LLM 模型，协处理器主要加速神经网络主体计算与采样后处理。Tokenizer、对话模板拼接、多模态衔接等逻辑仍由 host 侧（CPU/NPU）协助完成。

当前协处理器对模型输入输出本身没有强制格式绑定，但为了获得更优性能，RKNN3-Toolkit 的 LLM 推理流程对导出的 ONNX 输入、输出与配置文件有明确约束。本文档基于当前代码实现（重点参考 `examples/Qwen3/python/export_llm.py` 与 `py_utils/export_llm_helper.py`）进行说明。

## 导出脚本与产物说明

以 `examples/Qwen3/python/export_llm.py` 为例，LLM 导出通常会生成以下文件：

- **`*.onnx`**：LLM 主体 ONNX 模型（由 `causal_llm_to_onnx()` 导出）。
- **`*.config.pkl`**：板端运行所需配置（system/prompt 信息、chat_template、vocab_size、hidden_size、hf_config_json 以及可选量化参数 `q_params`）。
- **`*.tokenizer.gguf`**：Tokenizer 文件（由 `export_tokenizer()` 导出）。
- **`*.embed.bin`**：Embedding 权重（fp16，来自 `model.model.embed_tokens.weight`，由 `export_embed_weight()` 导出）。

说明：

- `export_llm.py` 中支持 `--quant` 触发外部 GRQ 量化流程（成功后导出路径切到 `grq_model_path`）。

## ONNX 输入输出约束（当前实现）

### 1) 常规 Causal LLM（如 Qwen3）

基础输入（必需）：

- **`input_ids`**：`[1, sequence]`，`int64`
- **`attention_mask`**：`[1, sequence]`，`float32`
- **`position_ids`**：`[1, sequence]`，`int64`

可选输入（按模型 `forward` 签名自动追加）：

- **`num_logits_to_keep`**：`[1]`，`int32`。当模型 `forward` 含 `logits_to_keep` 或 `num_logits_to_keep` 参数时自动插入，默认值为 `-1`（仅保留最后一个 token logits，降低冗余计算/内存开销）。

输出：

- **单输出 `output`**（logits）。


## KV Cache 约束

- RKNN3-Toolkit 会基于 Attention OP 自动构建并管理 KV Cache（含性能与量化优化），用户无需在 ONNX 中显式维护 KV Cache 输入输出。
- 导出前建议将配置中的 `use_cache=False`（示例脚本通过 `update_config(config, ['use_cache'], False)` 处理），确保导出的 ONNX 不含 KV Cache I/O。

## Shape 与导出行为说明

- 运行时推荐采用多组静态序列长度（如 prefill 的 `N` 与 decode 的 `1`）进行部署配置。
- 目前 `causal_llm_to_onnx()` 内部会将：
  - `args.prompt_size = 64`
  - `args.dynamic_shape = True`
  作为导出默认行为，ONNX 中会为序列维标记动态轴 `sequence`。
- 板端加载时可通过 `rknn.load_llm` 的 `seq_len`（例如 `[1, 128]`）指定实际运行所需的静态长度组合。

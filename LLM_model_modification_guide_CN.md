# LLM模型适配说明

对于 LLM 模型，协处理器主要用于加速 LLM 模型的神经网络主体部分、以及 LLM 模型的采样后处理。对于 Tokenizer、后处理及多模态模型之间的衔接耦合操作，目前需要借助于 host 端的 CPU/NPU 协助处理。



当前协处理器对模型输入输出的定义**无特殊限制**。但为了在硬件上达到**更优的推理性能**，根据协处理器的硬件特性，RKNN 构建了一套 LLM 推理接口及流程，该流程要求 **LLM 模型**在导出 onnx 时需要符合一定的要求，具体要求如下：



## 部署示例生成文件说明
本仓库主要基于 HuggingFace 上的经典模型，提供模型转换和板端部署示例。转换脚本主要包含`export_llm.py`、`export_vision.py`、`export_rknn.py` 三部分，对应的功能及生成的文件如下:

1.`export_llm.py` 从 HugggingFace 格式的 LLM文件导出 ONNX模型和相关配置文件，生成文件含义如下

- **model_llm.onnx**: 适配RKNN3-Toolkit的ONNX模型文件，其中默认使用外部grq量化算法，模型本身自带量化参数。当不使用外部grq量化算法时，可使用RKNN3-Toolkit工具在模型转换过程中进行量化。
- **model_llm.config.pkl**: 使用RKNN3-Toolkit转换LLM模型需要的配置文件，其中包括chat template、vocab size、hidden size等描述模型规格的参数，以及使用外部grq量化算法的相关信息。
- **model_llm.embed.bin**: 模型Embedding层的权重，采用Float16存储。当前Embedding层的操作集中在主控端，主控端基于 Token Ids 查询到对应的 Embeds 数据并传输给协处理器。当模型的 Embedding 层操作比较复杂时，可以没有 model_llm.embed.bin 文件，此时用户需要在应用业务逻辑中自行管理处理 Embedding 操作。
- **model_llm.tokenizer.gguf**: 模型Tokenizer文件。RKNN3板端部署时默认支持Tokenizer功能，需要使用对应Tokenizer文件进行初始化。当使用自定义Tokenizer时可忽略此文件。

2.export_vision.py支持从HuggingFace导出Vision的ONNX模型，具体包括：

- **model_vision.onnx**: 适配RKNN3-Toolkit的ONNX模型文件，不支持使用外部量化。

3.export_rknn.py支持将ONNX模型转换为RKNN，具体包括：

- **model.rknn**: RKNN模型文件。
- **model.weight**: RKNN模型的权重文件。




## ONNX 格式下LLM模型的输入输出限制及含义:

**模型输入:** 必须包含且只包含以下4个输入

- **input_ids**: 输入的 token id，维度为 [1, sequence]，数据类型为 int64。

- **attention_mask**: 输入的因果推理mask，维度为 [1, sequence]，数据类型为 float32。主要用于控制固定输入shape情况下、补齐空输入对推理的影响。
- **position_ids**: 配置输入的位置编码 id，控制 rope 的采样点。
- **logit_id_to_keep**: 保留对应 id 的 logits 输出结果，主要用于 prefill 阶段保留最后一个有效 logits 输出，减少冗余的计算、内存开销。

**模型输出**: 

- 输出为单个 logit 的输出。



## KV Cache:

- RKNN3-Toolkit 在加载 LLM 模型时，会自动根据 Attention op 构筑 KV cache 的内部管理逻辑，包含推理性能、cache量化等优化，**无需用户自行管理 KV cache**。功能启用时，要求 onnx 模型不能有 KV cache 输入输出，通常可将 llm 模型 config 中的 use_cache 配置为 False 进行模型导出，此时导出 onnx 模型不会有 KV cache 输入输出。



## 支持多种输入 shape:

- RKNN 暂不支持实时动态 shape，目前支持静态多组 shape，用户需提前指定模型应用中所需的 shape。例如在 LLM 模型的支持场景中，prefill 阶段需要采用长度为 N 的输入进行推理，以获取更快的推理性能；在 generate 阶段采用长度为 1 的输入进行因果推理。

- 通过在 rknn.load_llm 接口配置 seq_len 参数，例如配置 seq_len = [1, 64]，生成输入 token 长度为 64 的 prefill 模型、以及输入 token 长度为 1 的 decoder 模型

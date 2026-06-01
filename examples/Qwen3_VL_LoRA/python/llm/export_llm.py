
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from onnx import numpy_helper
import numpy as np
import onnx

from transformers import masking_utils
masking_utils.sdpa_mask = masking_utils.sdpa_mask_older_torch

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_qwen3_vl_quantize_dataset
from transformers import AutoConfig # transformers==4.57.0


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_qwen3_vl import Qwen3VLForConditionalGeneration # 增加 num_logits_to_keep 输入

TARGET_RESHAPE_NAME = "/language_model/Reshape_1"
NEW_SHAPE = np.array([-1, 1], dtype=np.int64)


class LanguageModelWithLMHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(self, input_ids, attention_mask=None, position_ids=None, deepstack_embeds0=None, deepstack_embeds1=None, deepstack_embeds2=None, logits_to_keep=0):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deepstack_visual_embeds0=deepstack_embeds0,
            deepstack_visual_embeds1=deepstack_embeds1,
            deepstack_visual_embeds2=deepstack_embeds2,
            use_cache=False,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.select(1, logits_to_keep).unsqueeze(1))
        return logits

prompt = "RKLLM"
chat_context = {
    "messages":[
        {
            "role": "user",
            "content": [
                {"type": "image",},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "add_generation_prompt": True,
}



def replace_initializer(graph, name, new_shape):
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer[i].CopyFrom(
                numpy_helper.from_array(new_shape, name)
            )
            print(f"[OK] Replaced initializer shape: {name}")
            return True
    return False


def replace_constant_node(graph, name, new_shape):
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    attr.t.CopyFrom(
                        numpy_helper.from_array(new_shape)
                    )
                    print(f"[OK] Replaced Constant node shape: {name}")
                    return True
    return False

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3-VL llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="./Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct-llm.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    if args.load_weight:
        kwargs['config'] = config
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32, 
            # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
            low_cpu_mem_usage=True, _attn_implementation="eager",
            trust_remote_code=True)
        print("Loaded Qwen3-VL model weights successfully.")
        if args.quan_dataset:
            gen_qwen3_vl_quantize_dataset(args.model_path, model.eval(), model.language_model.embed_tokens, 
                                          '../../../../datasets/MMBench/llm/dataset.json', 
                                          '../../data/llm/dataset.txt', 
                                          '../../data/llm/dataset_np')
    else:
        kwargs.pop('trust_remote_code', True)
        model = Qwen3VLForConditionalGeneration._from_config(config, **kwargs)

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if export_llm_dirname and not os.path.exists(export_llm_dirname):
        print(f"create export_llm_dirname: {export_llm_dirname}")
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    wrapped_model = LanguageModelWithLMHead(model)
    wrapped_model.eval()

    args.hidden_size = config.text_config.hidden_size
    # 导出这个 wrapper
    causal_llm_to_onnx(wrapped_model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.language_model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)

    # modify onnx for rknn compatibility
    MODEL_IN = args.export_llm_path
    print(f"[INFO] Loading ONNX: {MODEL_IN}")
    model = onnx.load(MODEL_IN, load_external_data=False)
    graph = model.graph

    modified = False

    for node in graph.node:
        if node.op_type == "Reshape" and node.name == TARGET_RESHAPE_NAME:
            print(f"[INFO] Found target Reshape: {node.name}")

            if len(node.input) != 2:
                raise RuntimeError("Unexpected Reshape input count")

            shape_input = node.input[1]

            if replace_initializer(graph, shape_input, NEW_SHAPE):
                modified = True
                break

            if replace_constant_node(graph, shape_input, NEW_SHAPE):
                modified = True
                break

            raise RuntimeError(
                f"Shape source not found for Reshape: {shape_input}"
            )

    if not modified:
        raise RuntimeError("Target Reshape node not modified")

    print(f"[INFO] Saving RKNN-compatible ONNX: {MODEL_IN}")
    onnx.save(model, MODEL_IN)

    print("[DONE] ONNX modification finished successfully")

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_qwen2_vl_quantize_dataset
from transformers import AutoConfig
from transformers.utils.versions import require_version

require_version(
    "transformers<4.52.0",
    "This code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_gme_qwen2vl import GmeQwen2VL # 增加 num_logits_to_keep 输入
#msg = f'<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>'
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

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export GmeQwen2VL llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="iic/gme-Qwen2-VL-2B-Instruct")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/GmeQwen2VL-llm.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    if args.load_weight:
        kwargs['config'] = config
        model = GmeQwen2VL.from_pretrained(args.model_path, **kwargs)
        if args.quan_dataset:
            gen_qwen2_vl_quantize_dataset(args.model_path, model.eval(), model.model.embed_tokens, '../../../../datasets/MMBench/llm/dataset.json', '../../data/llm/dataset.txt', '../../data/llm/dataset_np')
    else:
        kwargs.pop('trust_remote_code', True)
        model = GmeQwen2VL._from_config(config, **kwargs)

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
            os.makedirs(export_llm_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    # 0: the generation task.
    # 1: the embedding task.
    user_config = {"task_type": 1,"patch_size": 14, "spatial_merge_size": 2}
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', None, None, user_config)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)

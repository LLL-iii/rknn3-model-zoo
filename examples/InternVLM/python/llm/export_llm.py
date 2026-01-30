import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_internvl_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_internvl_quantize_dataset
from transformers import AutoConfig,AutoModel,AutoTokenizer

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
    parser = ArgumentParser(description="Export Qwen/Qwen2.5-VL llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="OpenGVLab/InternVL3-2B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="./InternVL3-2B-llm/Qwen2.5-VL-1_5B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    if args.quant:
        import torch
        if torch.cuda.is_available():
            # 量化数据需要调整
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'

            # 加载 InternVL3_2B Chat 模型
            Intern_model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
            # 生成量化数据（替代 gen_qwen2_5_vl_quantize_dataset）
            gen_internvl_quantize_dataset(
                model_path=args.model_path,
                model=Intern_model,
                tokenizer=tokenizer,
                dataset_path='../../../../datasets/MMBench/llm/dataset.json',
                dataset_out_path='../../data/llm/dataset.txt',
                dataset_out_path_np='../../data/llm/dataset_np',
                grq_data=True
            )
            if grq_quantize(args.model_path, '../../../../datasets/MMBench/llm/grq_inputs.json', grq_model_path, group=32) == True:
                args.model_path = grq_model_path
                print("GRQ quantization success!")
            else:
                print("GRQ quantization failed!")
                exit(1)
        else:
            print("cuda is unavailable, ignore the '--quant' parameter!")

    kwargs = {
        'trust_remote_code': True,
    }
    
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    update_config(config, ['_attn_implementation'], 'eager')
    if args.load_weight:
        kwargs['config'] = config
        if args.quant:
            if config.architectures[0] == 'Qwen2ForCausalLM':
                from transformers import Qwen2ForCausalLM
                model = Qwen2ForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
            elif config.architectures[0] == 'Qwen3ForCausalLM':
                from transformers import Qwen3ForCausalLM
                model = Qwen3ForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
            else:
                print("This architecture is not supported")
        else :
            Intern_model = AutoModel.from_pretrained(args.model_path, **kwargs).eval()
            model = Intern_model.language_model
        if args.quan_dataset and not args.quant:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
            # 生成量化数据（替代 gen_qwen2_5_vl_quantize_dataset）
            gen_internvl_quantize_dataset(
                model_path=args.model_path,
                model=Intern_model,
                tokenizer=tokenizer,
                dataset_path='../../../../datasets/MMBench/llm/dataset.json',
                dataset_out_path='../../data/llm/dataset.txt',
                dataset_out_path_np='../../data/llm/dataset_np',
            )
    else:
        update_config(config, ["use_cache"], False)
        Intern_model = AutoModel.from_config(config, **kwargs)
        model = Intern_model.language_model

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
            os.makedirs(export_llm_dirname)



    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_internvl_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)
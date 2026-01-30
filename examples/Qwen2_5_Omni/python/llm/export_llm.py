import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_qwen2_5_omni_quantize_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Qwen2_5OmniForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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

    parser = ArgumentParser(description="Export Qwen/Qwen2.5-Omni llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/Qwen2.5-Omni-3B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    new_llm_path = "Qwen2.5-Omni-3B-language"
    omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).thinker.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(new_llm_path)

    # del omni_model.visual
    del omni_model.audio_tower
    omni_model.save_pretrained(new_llm_path)
    os.system("cp config.json {}/".format(new_llm_path))


    if args.quant:
        import torch
        if torch.cuda.is_available():
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            gen_qwen2_5_omni_quantize_dataset(args.model_path, omni_model, 
                omni_model.model.embed_tokens, 
                '../../../../datasets/MMBench/llm/dataset.json', 
                '../../data/llm/dataset.txt', '../../data/llm/dataset_np', True)
            if grq_quantize(new_llm_path, '../../../../datasets/MMBench/llm/grq_inputs.json', grq_model_path, group=32) == True:
                new_llm_path = grq_model_path
                print("GRQ quantization success!")
            else:
                print("GRQ quantization failed!")
                exit(1)
        else:
            print("cuda is unavailable, ignore the '--quant' parameter!")

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(new_llm_path, **kwargs)
    update_config(config, ['use_cache'], False)
    if args.load_weight:
        if args.quan_dataset and not args.quant:
            gen_qwen2_5_omni_quantize_dataset(args.model_path, omni_model, 
                omni_model.model.embed_tokens, 
                '../../../../datasets/MMBench/llm/dataset.json', 
                '../../data/llm/dataset.txt', '../../data/llm/dataset_np')
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(new_llm_path, **kwargs)
    else:
        kwargs.pop('trust_remote_code', True)
        model = AutoModelForCausalLM.from_config(config, **kwargs)

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_llm_config(new_llm_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_dirname)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_quantize_dataset
from transformers import AutoModelForCausalLM, AutoConfig
from export_from_modelscope import export_glm_to_onnx
from modelscope import snapshot_download

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}
    

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export GLM llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="ZhipuAI/glm-edge-1.5b-chat")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../model/llm/glm-edge-1.5b-chat.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    args = parser.parse_args()

    args.model_path = snapshot_download(args.model_path)

    if args.quant:
        import torch
        if torch.cuda.is_available():
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            if grq_quantize(args.model_path, '../../../datasets/CMMLU/dataset.json', grq_model_path, group=32) == True:
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
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
        if args.quan_dataset and not args.quant:
            gen_quantize_dataset(args.model_path, model.model.embed_tokens, '../../../datasets/CMMLU/dataset.json', '../../../datasets/CMMLU/dataset.txt', '../../../datasets/CMMLU/dataset_np')
    else:
        model = AutoModelForCausalLM.from_config(config, **kwargs)

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
            os.makedirs(export_dirname)

    # Export llm to onnx
    # export_model_to_onnx(args.model_path, args.export_llm_path, opset_version=18, args=args)
    export_glm_to_onnx(model, args.export_llm_path, opset_version=18, args=args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer', use_modelscope=True)  # glm-edge 走 modelscope 下载

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_dirname)

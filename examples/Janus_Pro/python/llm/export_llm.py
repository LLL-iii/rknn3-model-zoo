import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_janus_pro_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, make_dataset_for_janus_pro
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor # please execute install_janus.sh

prompt = "RKLLM"
conversation = [
    {
        "role": "<|User|>", 
        "content": f"<image_placeholder>\n{prompt}",
        "images":["../../data/vision/img.jpg"],
    },

    {
        "role": "<|Assistant|>", 
        "content": "",
    },
]

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export deepseek-ai/Janus-Pro llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/Janus-Pro-1B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    ori_model_name = args.model_path
    if args.quant:
        import torch
        if torch.cuda.is_available():
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            make_dataset_for_janus_pro(args.model_path, '../../../../datasets/MMBench/llm/dataset.json', '../../data/llm/dataset.txt', '../../data/llm/dataset_np', grq_data=True)
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
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
        if not args.quant:
            model = model.language_model
        if args.quan_dataset and not args.quant:
            make_dataset_for_janus_pro(args.model_path, '../../../../datasets/MMBench/llm/dataset.json', '../../data/llm/dataset.txt', '../../data/llm/dataset_np')
    else:
        model = AutoModelForCausalLM.from_config(config, **kwargs)

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_janus_pro_llm_config(ori_model_name, args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', conversation, prompt)

    # Export tokenizer
    export_tokenizer(ori_model_name, os.path.splitext(args.export_llm_path)[0] + '.tokenizer', use_modelscope=args.modelscope)  # LlamaTokenizer

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)
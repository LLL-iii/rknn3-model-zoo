import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_quantize_dataset
from transformers import AutoModelForCausalLM, AutoConfig
from custom_hunyuan_mask import patch_hunyuan_model

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export HY-MT1.5-1.8B llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Tencent-Hunyuan/HY-MT1.5-1.8B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../model/llm/HY-MT1.5-1.8B.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    if args.quant:
        import torch
        if torch.cuda.is_available():
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            if grq_quantize(args.model_path, '../data/dataset.json', grq_model_path, group=32) == True:
                print("GRQ quantization success!")
                import shutil
                src = args.model_path + "/tokenizer_config.json"
                dst = grq_model_path + "/tokenizer_config.json"
                shutil.copy2(src, dst)
            else:
                print("GRQ quantization failed!")
                exit(1)
        else:
            grq_model_path = None
            print("cuda must be available")
    else:
        grq_model_path = None


    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
        if args.quan_dataset and not args.quant:
            gen_quantize_dataset(args.model_path, model.model.embed_tokens, '../data/dataset.json', '../data/dataset.txt', '../data/dataset_np')
    else:
        model = AutoModelForCausalLM.from_config(config, **kwargs)

    model = patch_hunyuan_model(model)  # 应用mask补丁
    model.eval()  # 必须设为eval模式

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
            os.makedirs(export_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path if grq_model_path is None else grq_model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer', use_modelscope=args.modelscope)

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_dirname)
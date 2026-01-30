import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_minicpm_3o_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, make_dataset_for_minicpm_v
from transformers import AutoConfig, AutoModel
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiniCPM_V_4.modeling_minicpmv import MiniCPMV

prompt = "RKLLM"
img_path = "../../data/demo.jpg"
image = Image.open(img_path).convert('RGB')
message = [{'role': 'user', 'content': [image, prompt]}]

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export openbmb/MiniCPM-V-4 llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default='openbmb/MiniCPM-V-4')
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/MiniCPM-V-4-llm.onnx")
    parser.add_argument("--max_position_embeddings", type=int, help="max position embeddings", required=False, default=8192)
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    ori_model_path = args.model_path
    if args.quant:
        import torch
        if torch.cuda.is_available():
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            make_dataset_for_minicpm_v(args.model_path, '../../../../datasets/MMBench/llm/dataset.json', '../../data/llm/dataset.txt', '../../data/llm/dataset_np', grq_data=True)
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
    config.max_position_embeddings = args.max_position_embeddings
    if args.load_weight:
        kwargs['config'] = config
        model = MiniCPMV.from_pretrained(args.model_path, **kwargs)
        if args.quan_dataset and not args.quant:
            make_dataset_for_minicpm_v(args.model_path, '../../../../datasets/MMBench/llm/dataset.json', '../../data/llm/dataset.txt', '../../data/llm/dataset_np')
    else:
        model = AutoModel(config)

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model.llm, args)

    # Export LLM configuration 
    export_minicpm_3o_llm_config(ori_model_path, args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', message, prompt)

    # Export tokenizer
    os.system("cp ../MiniCPM_V_4/tokenizer_config.json {}".format(ori_model_path))
    export_tokenizer(ori_model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.llm.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)
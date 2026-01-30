import os, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_internvl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, AutoModel


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export InternViT-300M-448px-V2_5 vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default='OpenGVLab/InternVL3-2B') # 模型下载链接：https://huggingface.co/OpenGVLab/InternVL3_5-2B/tree/main
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="./onnx/InternViT3-vision.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {   
        'device_map': "cpu",
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation'], 'eager')
    if args.load_weight:
        print("args.load_weight")
        model = AutoModel.from_pretrained(args.model_path, **kwargs).eval()
    else:
        print("not args.from_config")
        kwargs = {   
            'torch_dtype': torch.float32,  
            'trust_remote_code': True,
        }
        update_config(config, ["use_cache"], False)
        model = AutoModel.from_config(config, **kwargs)


    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)
    
    # export vision model
    export_internvl_vision(model, args)

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)
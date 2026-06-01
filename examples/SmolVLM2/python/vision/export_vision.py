import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_smolvl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForImageTextToText

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export HuggingFaceTB/SmolVLM2-500M-Instruct vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/SmolVLM2-500M-vision.onnx")
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
    update_config(config, ['_attn_implementation'], 'eager')
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, **kwargs)
    else:
        kwargs.pop('trust_remote_code', True)
        model = AutoModelForImageTextToText._from_config(config, **kwargs)

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
            os.makedirs(export_vision_dirname)

    # export vision model
    export_smolvl_vision(model, args) 

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)
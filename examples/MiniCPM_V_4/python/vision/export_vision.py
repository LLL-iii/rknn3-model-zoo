import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_minicpm3o_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, AutoModelForCausalLM


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export openbmb/MiniCPM-V-4 vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default='openbmb/MiniCPM-V-4')
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/MiniCPM-V-4-vision.onnx")
    parser.add_argument("--max_position_embeddings", type=int, help="max position embeddings", required=False, default=8192)
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
    config.max_position_embeddings = args.max_position_embeddings
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    else:
        model = AutoModelForCausalLM(config)

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    # export vision model
    export_minicpm3o_vision(model.float(), args)

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)

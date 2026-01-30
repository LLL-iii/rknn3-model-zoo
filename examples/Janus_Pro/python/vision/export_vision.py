import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_janus_pro_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor # please execute install_janus.sh


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export deepseek-ai/Janus-Pro vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/Janus-Pro-1B-vision.onnx")
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
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_config(config, **kwargs)

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    model.eval().float()

    export_janus_pro_vision(model, args)

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)
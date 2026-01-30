import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_qwen2_vl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig
from transformers.utils.versions import require_version

require_version(
    "transformers<4.52.0",
    "This code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_gme_qwen2vl import GmeQwen2VL

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export GmeQwen2VL vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="iic/gme-Qwen2-VL-2B-Instruct")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/GmeQwen2VL-vision.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=448)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=448)
    
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
        model = GmeQwen2VL.from_pretrained(args.model_path, **kwargs)
    else:
        kwargs.pop('trust_remote_code', True)
        model = GmeQwen2VL._from_config(config, **kwargs)

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
            os.makedirs(export_vision_dirname)

    # export vision model
    export_qwen2_vl_vision(model.visual, args) # 添加grid_thw对输入图片的patch

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)
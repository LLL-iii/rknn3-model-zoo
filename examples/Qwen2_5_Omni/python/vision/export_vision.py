import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_qwen2_omni_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen2.5-Omni vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/Qwen2.5-Omni-3B-vision.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_size", type=int, help="Input image size (e.g., 224, 384, 448). Must be a multiple of 28.", required=False, default=392)
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float32,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    if args.load_weight:
        kwargs['config'] = config
        kwargs['low_cpu_mem_usage'] = True
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **kwargs).thinker.eval()
    else:
        kwargs.pop('trust_remote_code', True)
        model = Qwen2_5OmniForConditionalGeneration._from_config(config, **kwargs).thinker.eval()

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    # export vision model
    export_qwen2_omni_vision(model.visual, args)

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)
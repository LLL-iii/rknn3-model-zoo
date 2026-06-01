import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_qwen2_vl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, Qwen3VLForConditionalGeneration # transformers==4.57.0


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3-VL vision configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="Qwen3-VL-2B-vision.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    
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
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, 
            # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
            low_cpu_mem_usage=True, _attn_implementation="eager",
            trust_remote_code=True)
    else:
        kwargs.pop('trust_remote_code', True)
        model = Qwen3VLForConditionalGeneration._from_config(config, **kwargs)

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if export_vision_dirname and not os.path.exists(export_vision_dirname):
        print(f"create export_vision_dirname: {export_vision_dirname}")
        os.makedirs(export_vision_dirname)

    patch_size = 16
    if config.vision_config.patch_size is not None:
        patch_size = config.vision_config.patch_size
    else:
        print(f"patch_size is None, use default value {patch_size}")

    # export vision model
    export_qwen2_vl_vision(model.visual, args, patch_size) # 添加grid_thw对输入图片的patch

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)

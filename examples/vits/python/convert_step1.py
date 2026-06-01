import sys
from rknn.api import RKNN
import cv2
import numpy as np
import os

DEFAULT_RKNN_PATH = '../vits_step1.rknn'
DEFAULT_QUANT = False

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk1820]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8',]:
            do_quant = True
        else:
            do_quant = False
    
    DEFAULT_RKNN_PATH = os.path.splitext(model_path)[0] + f"_{model_type}" + '.rknn'

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    
    rknn.config(target_platform=platform,
                input_attrs={'x': {'dtype': 'int32', 'layout': 'UNDEFINED'}},
                quantized_dtype = 'w4a16',
                quantized_method='group32',
                quantized_algorithm = 'normal',
                # profile_mode = True, # 逐层dump需要设置为 True,
                core_num=1)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print(f'--> Export rknn model to {output_path}')
    ret = rknn.export_rknn(output_path, save_ctx=True)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('Export done')
    
    print('done')
    
    # Release
    rknn.release()

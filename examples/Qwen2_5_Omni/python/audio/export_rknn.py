import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../../model/audio/Qwen2.5-Omni-3B-audio.onnx'
RKNN_MODEL = '../../model/audio/Qwen2.5-Omni-3B-audio.rknn'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen2.5-Omni-3B audio to RKNN model")
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    dynamic_shapes = [
        [[1, 128, 300], [1, 1, 300], [1, 150, 150]],
        [[1, 128, 600], [1, 1, 600], [1, 300, 300]],
    ]

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, core_num=args.core_num, dynamic_input=dynamic_shapes)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()


import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../../model/llm/FastVLM-llm.onnx'
LLM_CONFIG = '../../model/llm/FastVLM-llm.config.pkl'
RKNN_MODEL = '../../model/llm/FastVLM-llm.rknn'
DATASET_PATH = '../../data/llm/dataset.txt'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export FastVLM llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform='rk1820', 
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    rknn.build(do_quantization=True, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    #Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()


import numpy as np
import os
from rknn.api import RKNN,DEFAULT_RKNN_LLM_CONFIG

ONNX_MODEL = './Qwen3-VL-2B-llm.onnx'
LLM_CONFIG = './Qwen3-VL-2B-llm.config.pkl'
RKNN_MODEL = '../../model/llm_2B/Qwen3-VL-2B-llm.rknn'
DATASET_PATH = '../../data/llm/dataset.txt'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen/Qwen3-VL llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)
    if "2B" in args.onnx_path:
        dynamic_input = [[[1, 1],   [1, 1],   [1, 1],   [1, 1, 2048],[1, 1, 2048],[1, 1, 2048], [1]], 
                            [[1, 128], [1, 128], [1, 128], [1, 128, 2048], [1, 128, 2048], [1, 128, 2048], [1]]]
    elif "4B" in args.onnx_path:
        dynamic_input = [[[1, 1],   [1, 1],   [1, 1], [1, 1, 2560], [1, 1, 2560], [1, 1, 2560], [1]], 
                            [[1, 128], [1, 128], [1, 128], [1, 128, 2560], [1, 128, 2560], [1, 128, 2560], [1]]]

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['attention_config'][0]['mrope_type'] = 'Qwen3-VL'
    llm_config['attention_config'][0]['mrope_section'] = [24,20,20] # Please refer to config.json to configure this parameters. eg, https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
    llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'


    # pre-process config
    print('--> config model')
    rknn.config(
        target_platform = 'rk1820', dynamic_input = dynamic_input,
        quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32',llm_config=llm_config
    )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config, seq=[1,128])

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

    # Export rknn model
    print('--> Export rknn model')
    export_rknn_dirname = os.path.dirname(args.rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        os.makedirs(export_rknn_dirname, exist_ok=True)
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
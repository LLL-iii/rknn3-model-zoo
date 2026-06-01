import sys
from rknn.api import RKNN
import numpy as np
import os

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
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = model_path.replace('.onnx', '.rknn')

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()
    
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform, core_num=1,
                input_attrs={'cached_len_0': {'dtype': 'float32', 'layout': 'UNDEFINED'}, 
                             'cached_len_1': {'dtype': 'float32', 'layout': 'UNDEFINED'},
                             'cached_len_2': {'dtype': 'float32', 'layout': 'UNDEFINED'},
                             'cached_len_3': {'dtype': 'float32', 'layout': 'UNDEFINED'},
                             'cached_len_4': {'dtype': 'float32', 'layout': 'UNDEFINED'}, },
                # profile_mode=True,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path,
                        inputs=['x',
                                'cached_len_0',
                                'cached_len_1',
                                'cached_len_2',
                                'cached_len_3',
                                'cached_len_4',
                                'cached_avg_0',
                                'cached_avg_1',
                                'cached_avg_2',
                                'cached_avg_3',
                                'cached_avg_4',
                                'cached_key_0',
                                'cached_key_1',
                                'cached_key_2',
                                'cached_key_3',
                                'cached_key_4',
                                'cached_val_0',
                                'cached_val_1',
                                'cached_val_2',
                                'cached_val_3',
                                'cached_val_4',
                                'cached_val2_0',
                                'cached_val2_1',
                                'cached_val2_2',
                                'cached_val2_3',
                                'cached_val2_4',
                                'cached_conv1_0',
                                'cached_conv1_1',
                                'cached_conv1_2',
                                'cached_conv1_3',
                                'cached_conv1_4',
                                'cached_conv2_0',
                                'cached_conv2_1',
                                'cached_conv2_2',
                                'cached_conv2_3',
                                'cached_conv2_4'],

                        input_size_list=[[1, 103, 80],
                                        [2, 1],
                                        [4, 1],
                                        [3, 1],
                                        [2, 1],
                                        [4, 1],
                                        [2, 1, 384],
                                        [4, 1, 384],
                                        [3, 1, 384],
                                        [2, 1, 384],
                                        [4, 1, 384],
                                        [2, 192, 1, 192],
                                        [4, 96, 1, 192],
                                        [3, 48, 1, 192],
                                        [2, 24, 1, 192],
                                        [4, 96, 1, 192],
                                        [2, 192, 1, 96],
                                        [4, 96, 1, 96],
                                        [3, 48, 1, 96],
                                        [2, 24, 1, 96],
                                        [4, 96, 1, 96],
                                        [2, 192, 1, 96],
                                        [4, 96, 1, 96],
                                        [3, 48, 1, 96],
                                        [2, 24, 1, 96],
                                        [4, 96, 1, 96],
                                        [2, 1, 384, 30],
                                        [4, 1, 384, 30],
                                        [3, 1, 384, 30],
                                        [2, 1, 384, 30],
                                        [4, 1, 384, 30],
                                        [2, 1, 384, 30],
                                        [4, 1, 384, 30],
                                        [3, 1, 384, 30],
                                        [2, 1, 384, 30],
                                        [4, 1, 384, 30]])
                            
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
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path, save_ctx=True)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
        
    
    print('done')

    # Release
    rknn.release()
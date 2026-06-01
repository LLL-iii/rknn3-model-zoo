from rknn.api import RKNN


DEFAULT_ONNX_PATH = '../../../models/encoder_online.onnx'
DEFAULT_RKNN_PATH = '../../../models/encoder_online.rknn'
DEFAULT_QUANT = False

def parse_arg():
    model_path = DEFAULT_ONNX_PATH
    platform = 'rk1820'
    do_quant = DEFAULT_QUANT
    output_path = DEFAULT_RKNN_PATH
    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0]], std_values=[[1]], target_platform=platform,
                input_attrs={'x': {'dtype': 'float32', 'layout': 'NCHW'}},
                quantized_dtype='w4a16',
                quantized_algorithm='grq',
                quantized_method='group32',
                core_num=8,
                )
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
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    
    # Release
    rknn.release()


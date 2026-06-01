from rknn.api import RKNN

ONNX_MODEL = '../../models/text_projector/text_projection.onnx'
RKNN_MODEL = '../../models/text_projector/text_projection.rknn'
TARGET_PLATFORM = 'rk1820'
DO_QUANT = False
DATASET_PATH = None


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen3-TTS text_projector to RKNN model")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL, help="onnx model path")
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="output rknn model path")
    parser.add_argument('--platform', type=str, default=TARGET_PLATFORM, help='Target platform (e.g. rk1820)')
    parser.add_argument('--do_quant', type=bool, default=DO_QUANT, help='whether to do quantization')
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="model quantization dataset path")
    args = parser.parse_args()

    rknn = RKNN(verbose=False)

    print('--> config model')
    rknn.config(
        target_platform=args.platform,
        quantized_dtype='w4a16',
        quantized_algorithm='normal',
        quantized_method='group32',
        core_num=1,
    )
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=args.do_quant, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()

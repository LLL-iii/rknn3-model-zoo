import ctypes
# import faulthandler
# faulthandler.enable()

import os
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "paddleocr_lib")))

_OCR_API_LIB = None

def _load_ocr_lib():
    global _OCR_API_LIB
    if _OCR_API_LIB is None:
        lib_path = Path(__file__).resolve().parent / "paddleocr_lib/libpaddle_vl.so"
        _OCR_API_LIB = ctypes.CDLL(str(lib_path))
    return _OCR_API_LIB

patch_size = 14

class RKNNApp(object):
    def __init__(self, model_path, vision_model_name, mlpAR_model_name, llm_model_name, position_embedding_model, vision_core_mask, mlpAR_core_mask, llm_core_mask, model_width, model_height):
        lib = _load_ocr_lib()
        llm_model_path = os.path.join(model_path, llm_model_name + ".rknn")
        llm_weight_path = os.path.join(model_path, llm_model_name + ".weight")
        tokenizer_path = os.path.join(model_path, llm_model_name + ".tokenizer")
        embeding_path = os.path.join(model_path, llm_model_name + ".embed.bin")
        vision_model_path =  os.path.join(model_path, vision_model_name + ".rknn")
        vision_weight_path = os.path.join(model_path, vision_model_name + ".weight")
        mlpAR_model_path =  os.path.join(model_path, mlpAR_model_name + ".rknn")
        mlpAR_weight_path = os.path.join(model_path, mlpAR_model_name + ".weight")
        position_embedding_model_path = os.path.join(model_path, position_embedding_model)
        self.model_init = lib.init_model
        self.model_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32]
        self.model_init.restype = ctypes.c_int
        self.model_init(ctypes.c_char_p(vision_model_path.encode('utf-8')), ctypes.c_char_p(vision_weight_path.encode('utf-8')), ctypes.c_char_p(position_embedding_model_path.encode('utf-8')), ctypes.c_char_p(llm_model_path.encode('utf-8')), ctypes.c_char_p(llm_weight_path.encode('utf-8')), ctypes.c_char_p(tokenizer_path.encode('utf-8')), ctypes.c_char_p(embeding_path.encode('utf-8')), ctypes.c_char_p(mlpAR_model_path.encode('utf-8')), ctypes.c_char_p(mlpAR_weight_path.encode('utf-8')), ctypes.c_uint32(vision_core_mask), ctypes.c_uint32(mlpAR_core_mask), ctypes.c_uint32(llm_core_mask), ctypes.c_uint32(model_width), ctypes.c_uint32(model_height))

        self.get_result = lib.get_result
        self.get_result.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        self.get_result.restype = ctypes.c_int
        self.result = ctypes.c_char_p(None)

        self.model_run = lib.inference_model
        self.model_run.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.model_run.restype = ctypes.c_int
        
        self.model_release = lib.release_model
        self.model_release.argtypes = []
        self.model_release.restype = ctypes.c_int
        self.model_name = "RKNNApp"
        
    def run(self, image: str, prompt):
        ret = self.model_run(ctypes.c_char_p(image.encode('utf-8')), ctypes.c_char_p(prompt.encode('utf-8')))
        if ret != 0:
            print("inference model failed")
            return
        self.get_result(ctypes.byref(self.result))
        res = ctypes.string_at(self.result).decode('utf-8', errors='ignore')
        # print(res)
        return res
    
    def release(self):
        self.model_release()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Deploy OCR model") 
    parser.add_argument("--model_path", type=str, help="model path", required=False, default="./models/")
    parser.add_argument("--vision_model_name", type=str, help="vision model name", required=False, default="PaddleOCR-vision")
    parser.add_argument("--mlpAR_model_name", type=str, help="mlpAR model name", required=False, default="PaddleOCR-vision-mlp_AR")
    parser.add_argument("--llm_model_name", type=str, help="llm model name", required=False, default="PaddleOCR-llm")
    parser.add_argument("--position_embedding_model", type=str, help="position embedding model name", required=False, default="position_embedding_model.bin")
    parser.add_argument("--vision_core_mask", type=int, help="vision core mask", required=False, default=0xff)
    parser.add_argument("--mlpAR_core_mask", type=int, help="mlpAR core mask", required=False, default=0xff)
    parser.add_argument("--llm_core_mask", type=int, help="llm core mask", required=False, default=0xff)
    parser.add_argument("--image_path", type=str, help="image path", required=False, default="/userdata/ocr.jpg")
    parser.add_argument("--prompt", type=str, help="prompt", required=False, default="OCR:")
    parser.add_argument("--model_width", type=int, help="model width", required=False, default=504)
    parser.add_argument("--model_height", type=int, help="model height", required=False, default=504)
    args = parser.parse_args()

    app = RKNNApp(args.model_path, args.vision_model_name, args.mlpAR_model_name, args.llm_model_name, args.position_embedding_model, args.vision_core_mask, args.mlpAR_core_mask, args.llm_core_mask, args.model_width, args.model_height)
    
    # image_path = "/userdata/ocr.jpg"
    # res = app.run(args.image_path, args.prompt)
    # print(res)

    for i in range(3):
        image_path = f"/userdata/{i}.png"
        res = app.run(image_path, args.prompt)
        print(res)

    app.release()


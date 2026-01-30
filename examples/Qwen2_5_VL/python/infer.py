import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import numpy as np
import cv2
import torch
from rknn.api import RKNN
import argparse
from transformers import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)

HUGGINGFACE_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
EMBED_PATH = '../model/llm/Qwen2.5-VL-3B-llm.embed.bin'
VOCAB_SIZE = 151936
IMAGE_TOKEN_ID = 151655
SEQ_LEN = 128 # rknn中load_llm设置的seq_len
step2 = True
max_new_tokens = 128

def get_llm_input_ids(img_path, prompt, img_h, img_w):
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(HUGGINGFACE_MODEL)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    img = Image.open(img_path)
    img = img.resize((img_w, img_h), 3)

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt], images=[img], padding=True, return_tensors="np"
    )

    return inputs["input_ids"]


def llm_logitsprocessor(input_ids, logits, args={}):
    temperature = args.get('temperature', 0.000001)
    top_k = args.get('top_k', 1)
    top_p = args.get('top_p', 1.0)
    repetition_penalty = args.get('repeat_penalty', 1.05)
    do_sample = args.get('do_sample', False)

    warpers = [
        TemperatureLogitsWarper(temperature), 
        RepetitionPenaltyLogitsProcessor(repetition_penalty) if input_ids is not None else None,
        TopKLogitsWarper(top_k=top_k), 
        TopPLogitsWarper(top_p=top_p), 
        ]

    for warper in warpers:
        if warper is not None:
            logits = warper(input_ids=input_ids, scores=logits)

    probs = torch.softmax(logits, dim=-1)

    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)[0]
    else:
        next_token = torch.argmax(probs, dim=-1)

    return next_token.numpy()

def llm_inference(rknn, token_ids, input_embeds, embeds_data, positions, eos_token_ids, tokenizer):
    generate_ids = []

    input_seq_len = input_embeds.shape[1]

    rope_cache = rknn.query('QUERY_ROPE_CACHE')

    print('--> Prefill Inference')
    if SEQ_LEN >= input_seq_len:
        inputs_embeds = np.zeros((1, SEQ_LEN, input_embeds.shape[-1]), dtype=np.float32)
        attention_mask = np.zeros((1, SEQ_LEN), dtype=np.float32)
        inputs_embeds[:,:input_seq_len,:] = input_embeds
        attention_mask[:,:input_seq_len] = 1
        position_id = np.arange(SEQ_LEN, dtype=np.float32).reshape(-1, 1)
        mrope_inputs = np.concatenate(position_id*3, axis=1) 
        mrope_inputs[:input_seq_len,:] = positions[:input_seq_len,:]
        num_logits_to_keep = np.array([input_seq_len - 1], dtype=np.int32)
        attention_inputs, dynamic_idx = rknn.kvcache_controller.generate_kvcache_control_tensors(input_seq_len)
        prefill_inputs = [inputs_embeds, attention_mask, num_logits_to_keep, mrope_inputs] + rope_cache + attention_inputs[0]
        data_format    = ['nchw'] * len(prefill_inputs)
        prefill_logits = rknn.inference(prefill_inputs, data_format, accuracy_analysis=False)[0]
    else:
        attention_inputs, dynamic_idx = rknn.kvcache_controller.generate_kvcache_control_tensors(input_seq_len)
        for i, seq_len in enumerate(range(0, input_seq_len, SEQ_LEN)):
            inputs_embeds = np.zeros((1, SEQ_LEN, input_embeds.shape[-1]), dtype=np.float32)
            attention_mask = np.zeros((1, SEQ_LEN), dtype=np.float32)
            curr_len = min(SEQ_LEN, input_seq_len - seq_len)
            inputs_embeds[:,:curr_len,:] = input_embeds[:,seq_len:seq_len+curr_len,:]
            attention_mask[:,:curr_len] = 1
            position_id = np.arange(start=seq_len, stop=seq_len+SEQ_LEN, dtype=np.float32).reshape(-1, 1)
            mrope_inputs = np.concatenate((position_id, position_id, position_id), axis=1)
            mrope_inputs[:curr_len,:] = positions[seq_len:seq_len+curr_len,:]
            num_logits_to_keep = np.array([curr_len - 1], dtype=np.int32)
            prefill_inputs = [inputs_embeds, attention_mask, num_logits_to_keep, mrope_inputs] + rope_cache + attention_inputs[i]
            data_format    = ['nchw'] * len(prefill_inputs)
            prefill_logits = rknn.inference(prefill_inputs, data_format, accuracy_analysis=False)[0]

    next_token = llm_logitsprocessor(torch.from_numpy(token_ids), torch.from_numpy(prefill_logits).reshape(1, -1))
    print(f"--> First new token:{next_token}")
    generate_ids.append(next_token[0])


    print("--> Decoder inference")
    max_position_id = np.max(positions)
    print("max_position_id: ", max_position_id)

    from tqdm import tqdm
    inf_bar = tqdm(range(max_new_tokens), desc='I RKLLM KVCache Inference ', ncols=100)
    for i in inf_bar:
        if i == max_new_tokens - 1:
            continue
        input_ids = np.expand_dims(next_token, axis=0).astype(np.int64)
        inputs_embeds = embeds_data[input_ids].astype(np.float32)

        max_position_id += 1
        position_id = np.expand_dims(np.expand_dims(np.array([max_position_id]), axis=0), axis=0).astype(np.int64)
        position_id = np.concatenate((position_id, position_id, position_id), axis=0).reshape(1, -1)

        attention_mask = np.expand_dims(np.array([1]), axis=0).astype(np.float32)
        num_logits_to_keep = np.array([0]).astype(np.int32)

        token_ids = np.concatenate((token_ids, input_ids), axis=1)

        attention_inputs, dynamic_idx = rknn.kvcache_controller.generate_kvcache_control_tensors(1)

        decoder_inputs = [inputs_embeds, attention_mask, num_logits_to_keep, position_id] + rope_cache + attention_inputs[0]
        data_format    = ['nchw'] * len(decoder_inputs)

        decoder_logits = rknn.inference(decoder_inputs, data_format)[0]
        next_token = llm_logitsprocessor(torch.from_numpy(token_ids), torch.from_numpy(decoder_logits).reshape(1, -1))
        generate_ids.append(next_token[0])

        if next_token[-1] in eos_token_ids:
            print('LLM Inference has completed!')
            break
        if (i+1) % 10 == 0:
            print("generate_ids: ", generate_ids)
            response = tokenizer.decode(generate_ids, skip_special_tokens=True)
            print("--> Intermediate response:\n", response)
    
    return generate_ids

def img_preprocess(img_path, img_h,img_w):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    return img

def prune_model_img_process(img,img_h,img_w):
    img = cv2.imread(img)
    img = cv2.resize(img, (img_w, img_h)).transpose(2,0,1).reshape(1,3,img_h, img_w)
    img = np.float16(img)
    img[0,2,...] = (img[0,2,...] - 104.09)/70.3 
    img[0,1,...] = (img[0,1,...] - 116.74)/66.6 
    img[0,0,...] = (img[0,0,...] - 122.70)/68.5 
    patches = np.concatenate([img, img], axis=1)
    h = img.shape[2]
    w = img.shape[3]
    print(f"prune model img process input shape: {img.shape}, patches shape: {patches.shape}")
    patches = patches.reshape(1, 2, 3, h // 2 // 14, 2, 14, w // 2 // 14, 2, 14)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    feature = patches.reshape(1 * h // 14 * w // 14, 3 * 2 * 14 * 14)
    return feature

def generate_qwen2_5_vl_mrope_pos_ids(input_ids, image_grid_thw, image_token_id=151655):
    """
    :param input_ids: 一维 np.array, 包含文本和 IMAGE_TOKEN_ID
    :param image_grid_thw: 列表或数组，形状为 (num_images, 3)，如 [[t, h, w], ...]
    :param image_token_id: 图像占位符 ID (Qwen2.5-VL 默认为 151655)
    :return: shape 为 (seq_len, 3) 的 np.array
    """
    input_ids = np.array(input_ids).flatten()
    seq_len = len(input_ids)
    
    position_ids = np.zeros((3, seq_len), dtype=np.int64)
    
    img_mask = (input_ids == image_token_id)
    
    if not np.any(img_mask):
        # 纯文本情况
        ids = np.arange(seq_len)
        return np.stack([ids, ids, ids])

    curr_pos = 0       # input_ids 的扫描指针
    curr_st_idx = 0    # 逻辑位置计数器 (Sequence ID 偏移)
    img_count = 0      # 图像计数器
    
    while curr_pos < seq_len:
        if input_ids[curr_pos] != image_token_id:
            next_img_indices = np.where(img_mask[curr_pos:])[0]
            text_len = next_img_indices[0] if len(next_img_indices) > 0 else (seq_len - curr_pos)
            
            text_offsets = np.arange(text_len) + curr_st_idx
            position_ids[:, curr_pos : curr_pos + text_len] = text_offsets
            
            curr_pos += text_len
            curr_st_idx += text_len
        else:
            if img_count >= len(image_grid_thw):
                raise ValueError("图像 token 块数量多于 image_grid_thw 提供的参数数量")
            
            t, h, w = image_grid_thw[img_count]
            img_tokens_num = t * h * w
            
            t_grid = np.repeat(np.arange(t), h * w)
            h_grid = np.tile(np.repeat(np.arange(h), w), t)
            w_grid = np.tile(np.arange(w), t * h)
            
            position_ids[0, curr_pos : curr_pos + img_tokens_num] = t_grid + curr_st_idx
            position_ids[1, curr_pos : curr_pos + img_tokens_num] = h_grid + curr_st_idx
            position_ids[2, curr_pos : curr_pos + img_tokens_num] = w_grid + curr_st_idx
            
            curr_st_idx += max(t, h, w)
            curr_st_idx = position_ids[:, curr_pos : curr_pos + img_tokens_num].max() + 1
            
            curr_pos += img_tokens_num
            img_count += 1
            
    return position_ids.transpose(1,0)

class Qwen2_5_VL():
    def __init__(self, args):
        self.llm_model = args.llm_model
        self.vision_model = args.vision_model
        self.target = args.target
        self.device_id = args.device_id
        self.img = args.img
        self.prompt = args.prompt
        self.prune_mode = args.prune_mode
        self.img_h = args.img_h
        self.img_w = args.img_w


    def qwen_vl_llm_run(self, img_embeds):
        rknn = RKNN()
        if step2:
            rknn.load_rknn(self.llm_model, self.llm_model.replace(".rknn", ".weight"), load_ctx=True)
        else:
            rknn.load_rknn(self.llm_model, self.llm_model.replace(".rknn", ".weight"))

        if self.target is not None:
            print("--> Qwen2.5-VL-3B-Instruct LLM init runtime")
            rknn.init_runtime(target=self.target, core_mask=0xff, device_id=self.device_id)
        else:
            rknn.init_runtime()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)

        input_ids = get_llm_input_ids(self.img, self.prompt, self.img_h, self.img_w)

        embeds_data = np.fromfile(EMBED_PATH, dtype=np.float16).reshape(VOCAB_SIZE, -1)
        input_embeds = embeds_data[input_ids].astype(np.float32)

        img_mask = input_ids == IMAGE_TOKEN_ID
        input_embeds[img_mask] = img_embeds

        image_grid_thw = [
            [1, self.img_h//14//2, self.img_w//14//2],  # img1
            # [1, self.img_h//14//2, self.img_w//14//2],  # img2
        ]
        # 生成position ids
        position_ids = generate_qwen2_5_vl_mrope_pos_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            image_token_id=IMAGE_TOKEN_ID
        )

        print("--> Qwen2.5-VL-3B-Instruct LLM Inference")
        generate_ids = llm_inference(rknn, input_ids, input_embeds, embeds_data, position_ids, [tokenizer.eos_token_id], tokenizer)
        response = tokenizer.decode(generate_ids, skip_special_tokens=True)

        rknn.release()

        return response

    def qwen_vl_vision_run(self):
        rknn = RKNN()
        if step2:
            rknn.load_rknn(self.vision_model, self.vision_model.replace(".rknn", ".weight"), load_ctx=True)
        else:
            rknn.load_rknn(self.vision_model, self.vision_model.replace(".rknn", ".weight"))

        if self.target is not None:
            print("--> Qwen2.5-VL-3B-Instruct Vision init runtime")
            rknn.init_runtime(target=self.target, core_mask=0xff, device_id=self.device_id)
        else:
            rknn.init_runtime()
        if self.prune_mode == True:
            img = prune_model_img_process(self.img, self.img_h, self.img_w)
        else:
            img = img_preprocess(self.img, self.img_h, self.img_w)

        print("--> Qwen2.5-VL-3B-Instruct Vision Inference")
        outputs = rknn.inference(inputs=[img])
        rknn.release()

        return outputs[0]

    def run(self):
        img_embeds = self.qwen_vl_vision_run()
        response = self.qwen_vl_llm_run(img_embeds)

        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rknn model test')
    parser.add_argument('--llm_model', type=str, help="Qwen2.5-VL-3B-Instruct LLM model path", default='../model/llm/Qwen2.5-VL-3B-llm.rknn')
    parser.add_argument('--vision_model', type=str, help="Qwen2.5-VL-3B-Instruct Vision model path", default='../model/vision/Qwen2.5-VL-3B-vision.rknn')
    parser.add_argument('--target', '-t', type=str, help="target platform", default=None)
    parser.add_argument('--device_id', type=str, help="target device id", default=None)
    parser.add_argument('--img', type=str, help="input image", default='../data/vision/demo.jpg')
    parser.add_argument('--prompt', type=str, help="input prompt", default='请描述图片内容')
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=392)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=392)
    parser.add_argument("--no_prune_mode", dest="prune_mode", action="store_false", help="close prune mode", default=True)

    args = parser.parse_args()
    qwen_vl = Qwen2_5_VL(args)
    response = qwen_vl.run()

    print("--> Qwen2.5-VL-3B-Instruct demo result:")
    print(response)
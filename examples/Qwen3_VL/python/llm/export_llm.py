import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from transformers import masking_utils
masking_utils.sdpa_mask = masking_utils.sdpa_mask_older_torch

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_qwen3_vl_quantize_dataset
from transformers import AutoConfig # transformers==4.57.0


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel # 增加 num_logits_to_keep 输入



class LanguageModelWithLMHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.config = model.config
        self.device = model.device

    def forward(self, input_ids, attention_mask=None, position_ids=None, deepstack_embeds0=None, deepstack_embeds1=None, deepstack_embeds2=None, logits_to_keep=0):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deepstack_visual_embeds0=deepstack_embeds0,
            deepstack_visual_embeds1=deepstack_embeds1,
            deepstack_visual_embeds2=deepstack_embeds2,
            use_cache=False,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.select(1, logits_to_keep).unsqueeze(1))
        return logits

prompt = "RKLLM"
chat_context = {
    "messages":[
        {
            "role": "user",
            "content": [
                {"type": "image",},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3-VL llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="./Qwen3-VL-2B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    if args.quant:
        import torch
        if torch.cuda.is_available():
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.float32, 
                # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
                low_cpu_mem_usage=True, _attn_implementation="eager",
                trust_remote_code=True)
            gen_qwen3_vl_quantize_dataset(args.model_path, model.eval(), model.language_model.embed_tokens, 
                                            '../../../../datasets/MMBench/llm/dataset.json', 
                                            '../../data/llm/dataset.txt', 
                                            '../../data/llm/dataset_np',
                                            grq_data=True)
            if grq_quantize(args.model_path, '../../../../datasets/MMBench/llm/grq_inputs.json', grq_model_path, group=32) == True:
                print("GRQ quantization success!")
            else:
                print("GRQ quantization failed!")
                grq_model_path = None
                exit(1)
            del model
        else:
            grq_model_path = None
            print("cuda is unavailable, ignore the '--quant' parameter!")
    else:
        grq_model_path = None
            
    

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)

    if args.load_weight:
        kwargs['config'] = config
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True, _attn_implementation="eager",
            trust_remote_code=True)
        print("Loaded Qwen3-VL model weights successfully.")
        
        if grq_model_path is not None:
            text_model = Qwen3VLTextModel.from_pretrained(
                grq_model_path,
                torch_dtype=torch.float32, 
                low_cpu_mem_usage=True, _attn_implementation="eager",
                trust_remote_code=True)
            model.model.language_model = text_model
            wrapped_model = LanguageModelWithLMHead(model)
            wrapped_model.eval()
            wrapped_model.config.text_config = AutoConfig.from_pretrained(grq_model_path)
        else:
            if args.quan_dataset and not args.quant:
                gen_qwen3_vl_quantize_dataset(args.model_path, model.eval(), model.language_model.embed_tokens, 
                                            '../../../../datasets/MMBench/llm/dataset.json', 
                                            '../../data/llm/dataset.txt', 
                                            '../../data/llm/dataset_np')
            wrapped_model = LanguageModelWithLMHead(model)
            wrapped_model.eval()
    else:
        kwargs.pop('trust_remote_code', True)
        model = Qwen3VLForConditionalGeneration._from_config(config, **kwargs)
        wrapped_model = LanguageModelWithLMHead(model)
        wrapped_model.eval()

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if export_llm_dirname and not os.path.exists(export_llm_dirname):
        print(f"create export_llm_dirname: {export_llm_dirname}")
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    args.hidden_size = config.text_config.hidden_size
    
    # 导出这个 wrapper
    causal_llm_to_onnx(wrapped_model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)
    if grq_model_path is not None:
        import pickle
        with open(os.path.splitext(args.export_llm_path)[0] + '.config.pkl', "rb") as f:
            llm_config = pickle.load(f)
        llm_config["q_params"] = {
                'bits': wrapped_model.config.text_config.quantization_config['bits'],
                'sym': wrapped_model.config.text_config.quantization_config['sym'],
                'group_size': wrapped_model.config.text_config.quantization_config['group_size'],
            }
        with open(os.path.splitext(args.export_llm_path)[0] + '.config.pkl', "wb") as f:
            pickle.dump(llm_config, f)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.language_model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_llm_dirname)
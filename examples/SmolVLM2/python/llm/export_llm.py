import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_smol_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_smolvlm_quantize_dataset
from transformers import AutoModelForImageTextToText, AutoConfig
import torch


class LanguageModelWithLMHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.model.text_model
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0):
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.select(1, logits_to_keep).unsqueeze(1))
        return logits


# HuggingFace meta llama requires a license, please download the model from modelscope or execute download.sh

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export SmolVLM-500M-Instruct llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--quan_dataset", type=int, help="Whether generate quantization dataset, load weight must to True", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/SmolVLM2-500M-llm.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        # args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    # update_config(config, ['use_cache'], False)
    # update_config(config, ['_attn_implementation'], 'eager')
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, **kwargs)
        if args.quan_dataset:
            gen_smolvlm_quantize_dataset(args.model_path, model, model.model.text_model.embed_tokens, '../../../../datasets/MMBench/llm/dataset.json', '../../data/llm/dataset.txt', '../../data/llm/dataset_np')
    else:
        model = AutoModelForImageTextToText.from_config(config, **kwargs)

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
            os.makedirs(export_dirname)

    model.config.use_cache = False
    model.config._attn_implementation = 'eager'
    
    wrapped_model = LanguageModelWithLMHead(model)
    wrapped_model.eval()

    # Export llm to onnx
    causal_llm_to_onnx(wrapped_model, args)

    # Export LLM configuration 
    export_smol_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl')

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.text_model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_dirname)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_quantize_dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": False,
}

class SimpleWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")

    def forward(self, input_ids, attention_mask, position_ids, logits_to_keep):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logits_to_keep=logits_to_keep,
            output_hidden_states=True,
            return_dict=True
        )
        batch_scores = outputs.logits
        true_vector = batch_scores[:, :, self.token_true_id]
        false_vector = batch_scores[:, :, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        scores = torch.nn.functional.softmax(batch_scores, dim=1)[:,1]
        return scores

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3-Reranker llm configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../model/llm/Qwen3-Reranker-0.6B.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default="'../../../datasets/CMMLU/dataset.txt'")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    if args.quant:
        import torch
        if torch.cuda.is_available():
            from rknn.utils.grq import grq_quantize
            grq_model_path = os.path.dirname(args.export_llm_path)+'/grq'
            if grq_quantize(args.model_path, args.dataset_path, grq_model_path, group=32) == True:
                args.model_path = grq_model_path
                print("GRQ quantization success!")
            else:
                print("GRQ quantization failed!")
                exit(1)
        else:
            print("cuda is unavailable, ignore the '--quant' parameter!")
            
    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_config(config, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = SimpleWrapper(model)

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
            os.makedirs(export_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    # 0: the generation task.
    # 1: the Embedding task.
    # 2: the Reranker task.
    user_config = {"task_type": 2}
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', None, None, user_config)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer', use_modelscope=args.modelscope)

    # Export Reranker weight
    export_embed_weight(model.model.get_input_embeddings().weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_dirname)
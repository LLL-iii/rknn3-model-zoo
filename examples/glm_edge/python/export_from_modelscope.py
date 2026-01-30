#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to download GLM-Edge-1.5B-Chat model from ModelScope and convert to ONNX format
Fixed version to handle DynamicCache issue
"""

import torch
import os
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["TORCH_ONNX_USE_EXPERIMENTAL_EXPORT"] = "0"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from py_utils.export_llm_helper import check_gptq, register_bitwise_right_shift
    
def export_glm_to_onnx(model, output_path, opset_version=19, args=None):
    """
    Export GLM-Edge-1.5B-Chat model to ONNX format
    Fixed to handle DynamicCache by disabling cache during export
    """
    
    print(f"Model loaded successfully. Model type: {type(model)}")
    print(f"Vocabulary size: {model.config.vocab_size}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Number of key-value heads: {model.config.num_key_value_heads}")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a wrapper class to avoid DynamicCache issues
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, position_ids, logits_to_keep):
            # logits_to_keep = int(num_logits_to_keep.item())
            # print("num_logits_to_keep:")
            # print(num_logits_to_keep)
            
            # print("input_ids:")
            # print(input_ids)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                cache_position=None,
                num_logits_to_keep = logits_to_keep.item(),
            )
            return outputs.logits
    
    # Wrap the model
    wrapped_model = ModelWrapper(model)
    
    dynamic_axes = {
        'input_ids': {1: 'sequence'},
        'attention_mask': {1: 'sequence'},
        'position_ids': {1: 'sequence'},
        # 'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'}
        # 'output': {1: 'sequence'},
    }
    
    
    in_len = 64
    dummy_input = torch.zeros((1, in_len), dtype=torch.long)
    attention_mask = torch.ones((1, in_len), dtype=torch.float)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    inputs = (dummy_input, attention_mask, position_ids)
    
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["output"]
    
    logit_keep_keys = ['logits_to_keep', 'num_logits_to_keep']
    logit_keep_key  = None
    _forward_func = model.forward
    while hasattr(_forward_func, '__wrapped__'):
        _forward_func = _forward_func.__wrapped__

    print(_forward_func.__code__.co_varnames)
    
    for key in logit_keep_keys:
        if key in _forward_func.__code__.co_varnames:
            logit_keep_key = key
            break
    if logit_keep_key:
        # 只留最后一个 token 的 logits，减少计算量
        num_logits_to_keep = torch.tensor(1, dtype=torch.int32).reshape(1)
        # num_logits_to_keep = torch.tensor([1], dtype=torch.float32) 
        print(num_logits_to_keep)
        insert_nones = [None]* (_forward_func.__code__.co_varnames.index(logit_keep_key) - len(inputs) -1)
        inputs = (*inputs, num_logits_to_keep)
        input_names.append('num_logits_to_keep')
        
    if getattr(args, 'output_hidden_states', False) and 'output_hidden_states' in _forward_func.__code__.co_varnames:
        idx = _forward_func.__code__.co_varnames.index('output_hidden_states') - 1
        if idx < len(inputs):
            # 如果有 output_hidden_states 参数，则需要在输入中添加一个 None
            inputs = (*inputs[:idx], True, *inputs[idx+1:])
        else:
            inputs = (*inputs, (None)*(_forward_func.__code__.co_varnames.index('output_hidden_states') - len(inputs) - 1), True)

    if hasattr(model.config, 'quantization_config'):
        q_config = model.config.quantization_config
        if check_gptq(q_config.bits, q_config.group_size) == False:
            print("The GRQ model only supports:\n W4A16 grouped asymmetric and symmetric quantization with group sizes of 32, 64 or 128!!")
            exit(1)
        register_bitwise_right_shift()
    else:
        model.float()
    
    print(input_names)
    # Export the wrapped model
    torch.onnx.export(
        wrapped_model,
        inputs,
        # (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True,
    )
    
    print(f"Model successfully exported to: {output_path}")


def download_and_export():
    """
    Download the model from ModelScope and export to ONNX
    """
    # Model ID for GLM-Edge-1.5B-Chat on ModelScope
    model_id = "ZhipuAI/glm-edge-1.5b-chat"
    
    # Set output paths
    output_path = "./model/glm_edge_1.5b_chat.onnx"
    
    print(f"Downloading model {model_id} from ModelScope...")
    
    # Download model from ModelScope without cache_dir
    model_path = snapshot_download(
        model_id=model_id
    )
    
    print(f"Model downloaded to: {model_path}")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting ONNX export...")
    export_glm_to_onnx(model_path, output_path)
    print("ONNX export completed successfully!")


def main():
    download_and_export()


if __name__ == "__main__":
    main()
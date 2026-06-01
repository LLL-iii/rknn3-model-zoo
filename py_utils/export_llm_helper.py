import os
import torch
import numpy as np

import onnxruntime as ort

def register_bitwise_right_shift():
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx import symbolic_helper

    # 定义自定义的 symbolic 函数
    def custom_rshift(g, self, other):
        self_type = symbolic_helper._try_get_scalar_type(self).onnx_type()

        # 确保 `other` 转换为 `self` 的数据类型
        if symbolic_helper._try_get_scalar_type(other) != symbolic_helper._try_get_scalar_type(self):
            other = g.op("Cast", other, to_i=self_type)

        # 处理 uint8 类型
        if symbolic_helper._try_get_scalar_type(self) == torch.uint8:
            return g.op("BitShift", self, other, direction_s="RIGHT")

        # 其他类型按位右移逻辑
        two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
        if not symbolic_helper._is_fp(self):
            other = g.op("Cast", other, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        two_pow = g.op("Pow", two, other)
        two_pow = g.op("Cast", two_pow, to_i=self_type)
        # rshift = g.op("Div", self, two_pow)   # Div是向零取整, 但bitwise_right_shift是向下取整, 因此采用下面的方式实现向下取整

        # 向下取整实现
        div_result = g.op("Div", self, two_pow)
        sign_a = g.op("Min", g.op("Sign", self), g.op("Constant", value_t=torch.tensor(0)))  # 转换为 0 或 -1
        remainder = g.op("Mod", g.op("Abs", self), two_pow)
        has_remainder = g.op("Cast", g.op("Greater", remainder, g.op("Constant", value_t=torch.tensor(0))), to_i=self_type)
        rshift = g.op("Add", div_result, g.op("Mul", sign_a, has_remainder))

        return rshift

    # 注册自定义的 symbolic 函数
    register_custom_op_symbolic("aten::bitwise_right_shift", custom_rshift, 11)

def check_gptq(bit, group_size):
    if bit == 4 and group_size in [-1, 32, 64, 128]:
        return True
    return False


def causal_llm_to_onnx(model, args):
    import torch

    # Debug Parameter
    args.prompt_size = 64
    args.dynamic_shape = True

    model.eval()
    in_len = args.prompt_size

    dummy_input = torch.zeros((1, in_len), dtype=torch.long)
    attention_mask = torch.ones((1, in_len), dtype=torch.float)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    if hasattr(args, 'arch') and args.arch == "Qwen3-ASR":
        dummy_input = torch.zeros((1, in_len, args.hidden_size), dtype=torch.float)
        position_ids = torch.zeros((1, 1, in_len), dtype=torch.long)

    inputs = (dummy_input, attention_mask, position_ids)
    input_names = ["input_ids", "attention_mask", "position_ids"]
    if hasattr(args, 'arch') and args.arch == "Qwen3-ASR":
        input_names = ["input_embeds", "attention_mask", "position_ids"]
    dynamic_axes = {}
    if args.dynamic_shape:
        if hasattr(args, 'arch') and args.arch == "Qwen3-ASR":
            dynamic_axes.update({
                'input_embeds': {1: 'sequence'},
                'attention_mask': {1: 'sequence'},
                'position_ids': {2: 'sequence'},
            })
        else:
            dynamic_axes.update({
                'input_ids': {1: 'sequence'},
                'attention_mask': {1: 'sequence'},
                'position_ids': {1: 'sequence'},
            })

    # 获取 forward 参数
    forward_args = model.forward.__code__.co_varnames
    for i in range(3):
        name = f"deepstack_embeds{i}"
        if name in forward_args:
            embed = torch.randn(1, in_len, args.hidden_size, dtype=torch.float32)
            inputs = (*inputs, embed)
            input_names.append(name)
            dynamic_axes[name] = {1: "sequence"}

    output_names = ["output"]

    logit_keep_keys = ['logits_to_keep', 'num_logits_to_keep']
    logit_keep_key  = None
    _forward_func = model.forward
    while hasattr(_forward_func, '__wrapped__'):
        _forward_func = _forward_func.__wrapped__

    for key in logit_keep_keys:
        if key in _forward_func.__code__.co_varnames:
            logit_keep_key = key
            break
    if logit_keep_key:
        # 只留最后一个 token 的 logits，减少计算量
        num_logits_to_keep = torch.tensor(-1, dtype=torch.int32).reshape(1)
        insert_nones = [None]* (_forward_func.__code__.co_varnames.index(logit_keep_key) - len(inputs) -1)
        inputs = (*inputs, *insert_nones, num_logits_to_keep)
        input_names.append('num_logits_to_keep')


    if getattr(args, 'output_hidden_states', False) and 'output_hidden_states' in _forward_func.__code__.co_varnames:
        idx = _forward_func.__code__.co_varnames.index('output_hidden_states') - 1
        if idx < len(inputs):
            # 如果有 output_hidden_states 参数，则需要在输入中添加一个 None
            inputs = (*inputs[:idx], True, *inputs[idx+1:])
        else:
            inputs = (*inputs, *((None,)*(_forward_func.__code__.co_varnames.index('output_hidden_states') - len(inputs) - 1)), True)

    if hasattr(model.config, 'quantization_config'):
        q_config = model.config.quantization_config
        if check_gptq(q_config.bits, q_config.group_size) == False:
            print("GRQ model quantization not supported. Only W4A16 quantization grouped or channel asymmetric/symmetric with group_size in {32, 64, 128} or -1 (per-channel) is supported.")
            exit(1)
        register_bitwise_right_shift()
    else:
        model.float()

    # out = model(*inputs)
    # if len(out) != len(output_names):
    #     print(f"WARNING: output number not match, expect {len(output_names)}, got {len(out)}")
    #     print(f"WARNING: try only keep one output")
    #     output_names = output_names[:1]

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            args.export_llm_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    if False:
        ort.set_default_logger_severity(3)  # 设置 ONNX Runtime 日志级别为 WARNING 及以上
        # class MyLoggingHandler(ort.LoggingHandler):
        #     def __init__(self):
        #         super().__init__()
            
        #     def log(self, severity, category, logid, code_location, message):
        #         # 在这里自定义处理日志消息
        #         if severity < 3:  # 只记录 WARNING 及以上级别的日志
        #             print(f"[ONNX Runtime] {message}")

        # # 注册自定义日志处理器
        # ort.set_default_logger(MyLoggingHandler())

        # check onnx model result
        print("Checking ONNX model output...")
        sess_onnx = ort.InferenceSession(args.export_llm_path, providers=['CPUExecutionProvider'])
        valid_inputs = [v.numpy() if isinstance(v, torch.Tensor) else v for v in inputs if v is not None]
        input_feed   = {sess_onnx.get_inputs()[i].name: valid_inputs[i] for i in range(len(valid_inputs))}
        for k, v in input_feed.items():
            if isinstance(v, np.ndarray):
                print(f"Input {k}: {v.shape} {v.dtype} {v}")
            else:
                print(f"Input {k}: {type(v)} {v}")

        output_onnx  = sess_onnx.run(output_names, input_feed)
        output_name  = sess_onnx.get_outputs()[0].name
        cos_sim      = torch.cosine_similarity(torch.tensor(output_onnx[0]).reshape(1,-1), out[0].detach().cpu().reshape(1,-1), dim=-1)
        eluer_dist   = torch.dist(torch.tensor(output_onnx[0]).reshape(1,-1), out[0].detach().cpu().reshape(1,-1), p=2)
        abs_diff     = torch.abs(torch.tensor(output_onnx[0]) - out[0].detach().cpu())
        print(f"Cosine Similarity       : {cos_sim.item()}")
        print(f"Euclidean Distance      : {eluer_dist.item()}")
        print(f"Max Absolute Difference : {abs_diff.max().item()}")
        print("first 10 elements of output:")
        print(f"ONNX output: {output_onnx[0].reshape(-1)[:10]}")
        print(f"PyTorch output: {out[0].detach().cpu().numpy().reshape(-1)[:10]}")

        if True:
            save_dir = os.path.join(os.path.dirname(args.export_llm_path), "src_io")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for k, v in input_feed.items():
                if isinstance(v, np.ndarray):
                    np.save(os.path.join(save_dir, f"{k}.npy"), v)
                else:
                    with open(os.path.join(save_dir, f"{k}.txt"), 'w') as f:
                        f.write(str(v))
            np.save(os.path.join(save_dir, f"{output_name}_onnx.npy"), output_onnx[0])
            np.save(os.path.join(save_dir, f"{output_name}_torch.npy"), out[0].detach().cpu().numpy())
            print(f"ONNX model input/output saved to {save_dir}")

    print(f"Exported to {os.path.abspath(args.export_llm_path)}")

# disable attribute that may cause error while export onnx
def update_config(_config, _attr_names, _value):
    from transformers import PretrainedConfig

    for _attr in dir(_config):
        if _attr in _attr_names:
            setattr(_config, _attr, _value)
        elif isinstance(getattr(_config, _attr), PretrainedConfig):
            update_config(getattr(_config, _attr), _attr_names, _value)


def export_tokenizer(model_path, tokenizer_path):
    '''Export tokenizer from Hugging Face model to GGUF format.
    Args:
        model_path (str): Path or name of the Hugging Face model.
        tokenizer_path (str): Path to save the exported tokenizer in GGUF format.
    '''
    import subprocess

    # remote用于决定是否从远程下载模型文件,如果model_path以'.'、'/'或'~'开头，则remote为0，表示本地文件；否则为1，表示远程文件。
    if model_path.startswith(('.', '/', '~')):
        remote=0
    else:
        remote=1

    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CMD="python3 {}/../tokenizer/thirdparty/llama_vocab/convert_hf_to_gguf.py --vocab-only --outtype f16 --outfile {} {} {}".format(current_dir, tokenizer_path, "--remote" if remote == 1 else "", model_path)

    result = subprocess.run(
        CMD,
        shell=True,
        capture_output=True,
        text=True
    )

    # 检查命令是否成功执行
    if result.returncode != 0:
        print(result.stderr) 
        print(f"Tokenizer exported failed.")
    else:
        print(f"Tokenizer exported to {tokenizer_path}")


def export_embed_weight(weight, embed_path):
    '''Export embedding weight to float16 .bin.
    Args:
        weight(torch.Tensor): Embedding weight tensor.
        embed_path (str): Path to save the exported embedding weight.
    '''
    import torch

    if not isinstance(weight, torch.Tensor):
        raise TypeError("Weight must be a torch.Tensor")
    
    weight_fp16 = weight.detach().cpu().to(torch.float16).numpy()

    try:
        with open(embed_path, 'wb') as f:
            weight_fp16.tofile(f)
        print(f"Embedding weight exported to {embed_path}")
    except Exception as e:
        print(f"Failed to export embedding weight: {str(e)}")

def split_chat_template_prompt(chat_template, chat_context, prompt="RKLLM"):
    from jinja2 import Template

    tmpl = Template(chat_template)

    chat_template = tmpl.render(**chat_context)

    prompt_prefix, prompt_postfix = chat_template.split(prompt)

    system_prompt = ""
    if 'system' in prompt_prefix:
        sys_idx = prompt_prefix.find("system")
        start_str = prompt_prefix[:sys_idx]
        second_start_idx =  prompt_prefix.find(start_str, sys_idx)
        system_prompt = prompt_prefix[:second_start_idx]

        prompt_prefix = prompt_prefix[second_start_idx:]

    return system_prompt, prompt_prefix, prompt_postfix


def export_internvl_config(model_path, config_path, chat_context=None, prompt="RKLLM"):
    """
      - 读取模型与分词器
      - 用 tokenizer.chat_template 渲染 chat_context（补齐 tools/add_generation_prompt 默认值）
      - 仅拼接文本片段（忽略图片），提取 system/prefix/postfix
      - 汇总并持久化导出配置（含量化可选项）

    参数：
      model_path:  模型路径/名称
      config_path: 导出 pkl 路径
      chat_context: {"messages":[{"role":"user","content": str|list|None}, ...],
                     "tools":[], "add_generation_prompt":bool} 结构
      prompt:      用于切分的最后一次出现的标记（默认 "RKLLM"）
    """
    import pickle
    from copy import deepcopy
    from jinja2 import Template
    from transformers import AutoConfig, AutoTokenizer

    # ---------- 读取配置/分词器 ----------
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ---------- 预处理上下文 & 模板渲染（若无模板则直接置空） ----------
    system_prompt = prompt_prefix = prompt_postfix = chat_template = ""
    if getattr(tokenizer, "chat_template", None):
        chat_template = tokenizer.chat_template

        # 规范化 chat_context
        ctx = deepcopy(chat_context) if chat_context is not None else {}
        msgs_in = ctx.get("messages", [])

        def _content_to_str(content):
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(c.get("text", ""))
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            return str(content)

        messages = [{"role": m.get("role", "user"),
                     "content": _content_to_str(m.get("content", ""))}
                    for m in msgs_in]

        rendered = Template(chat_template).render(
            messages=messages,
            tools=ctx.get("tools", []),
            add_generation_prompt=bool(ctx.get("add_generation_prompt", False)),
        )

        # 提取 system（Qwen 风格 <|im_start|>system ... <|im_end|>）
        sys_start, im_end = "<|im_start|>system\n", "<|im_end|>"
        sidx = rendered.find(sys_start)
        if sidx != -1:
            sidx += len(sys_start)
            eidx = rendered.find(im_end, sidx)
            if eidx != -1:
                system_prompt = rendered[sidx:eidx]

        # 按最后一次出现的 prompt 切分
        pidx = rendered.rfind(prompt)
        if pidx == -1:
            prompt_prefix, prompt_postfix = rendered, ""
        else:
            prompt_prefix = rendered[:pidx]
            prompt_postfix = rendered[pidx + len(prompt):]

    # ---------- 汇总导出 ----------
    if not hasattr(config, "llm_config"):
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
    else:
        vocab_size = config.llm_config.vocab_size
        hidden_size = config.llm_config.hidden_size
        
    llm_cfg = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
    }
    grq_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_cfg["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_cfg, f)

    # 可选：打印关键信息（便于调试）
    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])
    print(f"Model configuration exported to {config_path}")


def export_llm_config(model_path, config_path, chat_context, prompt, user_config=None):
    from transformers import AutoConfig, AutoTokenizer
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(tokenizer.chat_template )

    if tokenizer.chat_template is not None and chat_context is not None and prompt is not None:
        try:
            system_prompt, prompt_prefix, prompt_postfix = split_chat_template_prompt(tokenizer.chat_template, chat_context, prompt)
        except Exception as e:
            system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
            prompt_prefix  = "<|im_start|>user\n";
            prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
        chat_template = tokenizer.chat_template
    else:
        system_prompt, prompt_prefix, prompt_postfix = "", "", ""
        chat_template = ""

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])
    
    vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.text_config.vocab_size
    hidden_size = config.hidden_size if hasattr(config, "hidden_size") else config.text_config.hidden_size

    hf_config_json = json.dumps(config.to_dict())
    if user_config is not None:
        merged_config = {**config.to_dict(), **user_config}
        hf_config_json = json.dumps(merged_config)

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "hf_config_json": hf_config_json,
    }
    grq_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")



def export_smol_llm_config(model_path, config_path):
    from transformers import AutoConfig, AutoTokenizer
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("chat_template:\n", repr(tokenizer.chat_template)[1:-1])

    system_prompt = "<|im_start|>System: You are a useful assistant for concise replies.<end_of_utterance>\n"
    prompt_prefix = "User: "
    prompt_postfix = "<end_of_utterance>\nAssistant:"
    chat_template = ""
    
    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])
    
    vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.text_config.vocab_size
    hidden_size = config.hidden_size if hasattr(config, "hidden_size") else config.text_config.hidden_size

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }
    grq_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")


def export_janus_pro_llm_config(model_path, grq_model_path, config_path, conversation, prompt="RKLLM"):
    from janus.models import VLChatProcessor
    from transformers import AutoConfig
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    text_prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation, system_prompt=VLChatProcessor.system_prompt,
    )

    assert "<|User|>" in text_prompt, "Janus_Pro conversation not include <|User|> "

    system_prompt, _ = text_prompt.split("<|User|>")
    prompt_idx = text_prompt.find("<|User|>")
    prompt_prefix, prompt_postfix = text_prompt[prompt_idx:].split(prompt)

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": "" if tokenizer.chat_template is None else tokenizer.chat_template,
        "vocab_size": config.language_config.vocab_size,
        "hidden_size": config.language_config.hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }

    grq_config = AutoConfig.from_pretrained(grq_model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")

def export_minicpm_3o_llm_config(model_path, grq_model_path, config_path, message, prompt="RKLLM"):
    from PIL import Image
    from transformers import AutoProcessor, AutoConfig
    import json
    from copy import deepcopy
    import pickle
    import re

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    msgs_list = [message]
    images_list = [None]
    prompts_lists = []
    input_images_lists = []
    for image, msgs in zip(images_list, msgs_list):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)
        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)
        prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
        input_images_lists.append(images)

    pattern = "(<image>./</image>)"
    
    final_texts = []
    for index, text in enumerate(prompts_lists):
        image_tags = re.findall(pattern, text)
        text_chunks = text.split(pattern)
        final_text = ""
        for i in range(len(image_tags)):
            final_text = final_text + text_chunks[i] + processor.image_processor.get_slice_image_placeholder(input_images_lists[i][0].size, i,)
        final_text += text_chunks[-1]
        final_texts.append(final_text)

    system_prompt = ""
    prompt_prefix, prompt_postfix = final_texts[0].split(prompt)
    if 'system' in prompt_prefix:
        sys_idx = prompt_prefix.find("system")
        start_str = prompt_prefix[:sys_idx]
        second_start_idx =  prompt_prefix.find(start_str, sys_idx)
        system_prompt = prompt_prefix[:second_start_idx]

        prompt_prefix = prompt_prefix[second_start_idx:]

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": processor.tokenizer.chat_template,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }

    grq_config = AutoConfig.from_pretrained(grq_model_path, trust_remote_code=True)
    if hasattr(grq_config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': grq_config.quantization_config['bits'],
            'sym': grq_config.quantization_config['sym'],
            'group_size': grq_config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")

def export_minicpm_v_llm_config(model_path, config_path, conversation, prompt="RKLLM"):
    from transformers import AutoConfig, AutoTokenizer
    import pickle
    import json

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # pip install peft

    # 来自modeling_minicpmv中MiniCPMV类的chat方法
    text_prompt = ''
    for i, msg in enumerate(conversation):
        role = msg['role']
        content = msg['content']
        assert role in ['user', 'assistant']
        if i == 0:
            assert role == 'user', 'The role of first msg should be user'
            content = tokenizer.im_start + tokenizer.unk_token * config.query_num + tokenizer.im_end + '\n' + content
        text_prompt += '<用户>' if role=='user' else '<AI>'
        text_prompt += content
    text_prompt += '<AI>'

    system_prompt = ""
    prompt_prefix, prompt_postfix = text_prompt.split(prompt)

    print("system_prompt:\n", repr(system_prompt)[1:-1])
    print("prompt_prefix:\n", repr(prompt_prefix)[1:-1])
    print("prompt_postfix:\n", repr(prompt_postfix)[1:-1])

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": tokenizer.chat_template,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "hf_config_json": json.dumps(config.to_dict()),
    }

    if hasattr(config, 'quantization_config'):
        llm_config["q_params"] = {
            'bits': config.quantization_config['bits'],
            'sym': config.quantization_config['sym'],
            'group_size': config.quantization_config['group_size'],
        }

    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)

    print(f"Model configuration exported to {config_path}")

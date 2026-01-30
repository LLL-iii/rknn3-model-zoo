# custom_hunyuan_mask.py
import torch

def custom_create_causal_mask(
    input_ids_shape,
    dtype,
    device,
    past_key_values_length=0,
    padding_mask=None,
):
    batch_size, seq_length = input_ids_shape
    # 基础因果mask（tril生成下三角矩阵）
    causal_mask = torch.ones((seq_length, seq_length + past_key_values_length), dtype=dtype, device=device)
    causal_mask = torch.tril(causal_mask).view(1, 1, seq_length, seq_length + past_key_values_length)
    
    # 处理padding mask（改用广播，移除vmap）
    if padding_mask is not None:
        # 调整shape: [batch, seq_len] -> [batch, 1, seq_len, seq_len+past]
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        padding_mask = padding_mask.expand(-1, 1, seq_length, -1)
        causal_mask = causal_mask * padding_mask
    
    return causal_mask

# 通用补丁函数：适配所有Hunyuan-MT模型（无需导入具体类）
def patch_hunyuan_model(model):
    """
    动态替换Hunyuan模型的forward方法，重写mask生成逻辑
    :param model: AutoModelForCausalLM加载的Hunyuan模型实例
    :return: 打补丁后的模型
    """
    # 保存原始forward方法
    if hasattr(model, 'original_forward'):
        return model  # 避免重复打补丁
    
    # 针对Hunyuan模型的结构，替换model.model的forward（核心编码器）
    original_model_forward = model.model.forward
    
    def patched_model_forward(*args, **kwargs):
        """重写model.model的forward，替换mask生成"""
        # 提取输入参数（兼容位置参数和关键字参数）
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        past_key_values_length = kwargs.get("past_key_values_length", 0)
        
        # 仅当有attention_mask时，替换为自定义causal_mask
        if attention_mask is not None and input_ids is not None:
            try:
                causal_mask = custom_create_causal_mask(
                    input_ids_shape=input_ids.shape,
                    dtype=model.dtype,
                    device=input_ids.device,
                    past_key_values_length=past_key_values_length,
                    padding_mask=attention_mask,
                )
                # 替换kwargs中的attention_mask为自定义causal_mask
                kwargs["attention_mask"] = causal_mask
            except Exception as e:
                print(f"Mask生成替换警告: {e}，使用原始mask逻辑")
        
        # 调用原始forward
        return original_model_forward(*args, **kwargs)
    
    # 替换模型核心编码器的forward
    model.model.forward = patched_model_forward
    # 标记已打补丁
    model.original_forward = model.forward
    return model


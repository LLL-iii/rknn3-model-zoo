import os
import json
import numpy as np
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
import safetensors.torch as st
import re


def load_sharded_safetensors(model_dir):
    """
    加载分片的 safetensors 文件并合并成一个完整的 state_dict
    修复了路径中包含 '-' 时的分片编号解析问题
    
    Args:
        model_dir: 模型文件所在目录
    
    Returns:
        合并后的完整 torch 字典
    """
    file_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(file_path):
        print(f"找到完整模型文件: {file_path}，直接加载...")
        try:
            state_dict = st.load_file(file_path)
            print(f"成功加载完整模型，共 {len(state_dict)} 个参数")
            return state_dict
        except Exception as e:
            print(f"加载完整模型文件失败: {e}，尝试加载分片文件...")

    # 1. 查找所有分片的 safetensors 文件
    safetensors_files = []
    for file in os.listdir(model_dir):
        # 只匹配分片文件（包含 of- 且是 safetensors 格式）
        if file.endswith('.safetensors') and 'of-' in file:
            safetensors_files.append(os.path.join(model_dir, file))
    
    # 2. 安全的分片编号提取逻辑（重点修复）
    def extract_shard_number(file_path):
        """从文件路径中提取分片编号（如 00001）"""
        # 先提取纯文件名
        filename = os.path.basename(file_path)
        # 方法1：使用正则表达式匹配数字分片（推荐）
        match = re.search(r'-(\d+)-of-', filename)
        if match:
            return int(match.group(1))
        # 方法2：备用分割方式（兼容不同命名格式）
        parts = filename.split('-')
        for part in parts:
            if part.isdigit() and len(part) >= 4:  # 匹配 4 位以上的数字分片号
                return int(part)
        return 0  # 默认值
    
    # 按分片编号排序
    safetensors_files.sort(key=extract_shard_number)
    
    # 3. 加载并合并所有分片
    full_state_dict = {}
    for idx, file_path in enumerate(safetensors_files, 1):
        print(f"Loading shard {idx}/{len(safetensors_files)}: {file_path}")
        try:
            # 加载单个分片
            state_dict = st.load_file(file_path)
            # 合并到完整字典中
            full_state_dict.update(state_dict)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}, error: {e}")
            continue
    
    print(f"成功加载 {len(full_state_dict)} 个参数")
    return full_state_dict


class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size

class qwen3_asr_cnn(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )

    def forward(self, x):
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        b, c, f, t = x.size()
        x = self.conv_out(x.permute(0, 3, 1, 2).view(b, t, c * f))
        return x


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3ASRAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output
    

class Qwen3ASRAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class qwen3_asr_audio(nn.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )

        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList([Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def forward(self, x):
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        b, c, f, t = x.size()
        x = self.conv_out(x.permute(0, 3, 1, 2).view(b, t, c * f))
        positional_embedding = (
            self.positional_embedding.positional_embedding[: x.shape[1], :]
            .unsqueeze(0)
            .to(x.dtype)
        )
        x = x + positional_embedding
        x = x.view(-1, config.d_model)
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(x)
            x = layer_outputs[0]
        hidden_states = self.ln_post(x)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3 audio encoder onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--export_encoder_path", type=str, help="export audio encoder onnx model path", required=False, default="../../../models/encoder.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()
    
    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    save_dir = os.path.dirname(args.export_encoder_path)
    os.makedirs(save_dir, exist_ok=True)

    cnn_attn_state_dict = {}
    llm_state_dict = {}
    n1 = len("thinker.audio_tower.")
    n2 = len("thinker.model.")
    state_dict = load_sharded_safetensors(args.model_path)
    print("total load key num:", len(state_dict.keys()))
    for k in state_dict.keys():
        if k.find("thinker.audio_tower") > -1:
            cnn_attn_state_dict[k[n1:]] = state_dict[k]
        else:
            llm_state_dict[k[n2:]] = state_dict[k]
    print("cnn_attn_state_dict key num:", len(cnn_attn_state_dict.keys()))
    print("llm_state_dict key num:", len(llm_state_dict.keys()))

    config_path = os.path.join(args.model_path, "config.json")
    fp = open(config_path, "r")
    config_json = json.load(fp)
    fp.close()

    kwargs = {}
    for k in config_json['thinker_config']['audio_config'].keys():
        kwargs[k] = config_json['thinker_config']['audio_config'][k]
    config = Qwen3ASRAudioEncoderConfig(**kwargs)
    print(config)

    qwen3_asr_audio_model = qwen3_asr_audio(config)
    qwen3_asr_audio_model.load_state_dict(cnn_attn_state_dict, strict=True)
    qwen3_asr_audio_model.eval()

    x = torch.randn(8, 1, 128, 100, dtype=torch.float32)
    torch.onnx.export(
        qwen3_asr_audio_model,
        (x),
        args.export_encoder_path,
        opset_version=18,
        input_names=["x"],
        output_names=["h"],
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
    )


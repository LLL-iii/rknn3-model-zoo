import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
import sys

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(SCRIPT_DIR)

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)


from Qwen3_TTS.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration


def resolve_path(path, base_dir=PYTHON_ROOT):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_tensor_fp16_raw(tensor, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tensor_fp16 = tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()
    tensor_fp16.numpy().tofile(out_path)
    print(
        "[OK] Saved fp16 raw: {} | shape={} dtype={} size={:.2f} MB".format(
            out_path,
            tuple(tensor.shape),
            tensor_fp16.dtype,
            os.path.getsize(out_path) / 1024.0 / 1024.0,
        )
    )


def concat_embedding_weights_fp16(emb_layers):
    weights = []
    for i, emb in enumerate(emb_layers):
        if not hasattr(emb, "weight"):
            raise TypeError("Element {} is not an nn.Embedding-like module.".format(i))
        weights.append(emb.weight.detach())
    return torch.cat(weights, dim=0).to(dtype=torch.float16, device="cpu").contiguous()


def build_model(args):
    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)
    else:
        args.model_path = resolve_path(args.model_path)

    config = Qwen3TTSConfig.from_pretrained(args.model_path)
    config.use_cache = False
    config._attn_implementation = args.attn_implementation

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation=args.attn_implementation,
    )
    return model.eval()


def export_embeds(model, args):
    args.export_dir = resolve_path(args.export_dir)
    ensure_dir(args.export_dir)

    talker = model.talker

    text_emb = talker.get_text_embeddings().weight
    print("text_embeddings: shape={} dtype={}".format(tuple(text_emb.shape), text_emb.dtype))
    save_tensor_fp16_raw(text_emb, os.path.join(args.export_dir, "talker_text_embed.fp16.bin"))

    input_emb = talker.get_input_embeddings().weight
    print("input_embeddings: shape={} dtype={}".format(tuple(input_emb.shape), input_emb.dtype))
    save_tensor_fp16_raw(input_emb, os.path.join(args.export_dir, "talker_input_embed.fp16.bin"))

    code_emb_layers = talker.code_predictor.get_input_embeddings()
    code_emb_cat_fp16 = concat_embedding_weights_fp16(code_emb_layers)
    codec_out_path = os.path.join(args.export_dir, "codec_embed.fp16.bin")
    code_emb_cat_fp16.numpy().tofile(codec_out_path)
    print(
        "[OK] Saved fp16 raw: {} | shape={} dtype={} size={:.2f} MB".format(
            codec_out_path,
            tuple(code_emb_cat_fp16.shape),
            code_emb_cat_fp16.dtype,
            os.path.getsize(codec_out_path) / 1024.0 / 1024.0,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    saved_files = tokenizer.save_pretrained(args.export_dir)
    for saved_file in saved_files:
        if os.path.basename(saved_file) != "tokenizer.json" and os.path.exists(saved_file):
            os.remove(saved_file)
    print("[OK] Tokenizer saved to: {}".format(os.path.join(args.export_dir, "tokenizer.json")))

    print("\nAll embed exports done.")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen3-TTS embed files")
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path or name",
        required=False,
        default="../../../../CKPT/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        help="export embed directory",
        required=False,
        default="../models/embeds",
    )
    parser.add_argument("--modelscope", action="store_true", help="Whether download model from www.modelscope.cn")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        help="attention implementation passed into the model loader",
        required=False,
        default="eager",
    )

    args = parser.parse_args()
    model = build_model(args)
    export_embeds(model, args)

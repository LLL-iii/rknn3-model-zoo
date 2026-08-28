#!/usr/bin/env python3
"""导出 HuggingFace tokenizer 目录，供 RKNN3 新版 Tokenizer / ChatTemplate 从目录加载。

新版 C++ 端加载约定：
  - Tokenizer(model_dir)    -> 需要 <dir>/tokenizer.json（BPE）；存在 tokenizer.model 则走 SPM
  - ChatTemplate(model_dir) -> 优先 <dir>/chat_template.jinja，否则读
                               <dir>/tokenizer_config.json 的 chat_template 字段

用法：
  python3 export_tokenizer_hf.py <model_path> <output_dir>

<model_path> 可以是本地模型目录，也可以是 HuggingFace hub 模型名（如 Qwen/Qwen2.5-0.5B-Instruct）。
远程模型用 transformers.AutoTokenizer 下载并 save_pretrained（默认走 hf-mirror.com 镜像，
与 tokenizer/zoo_test.py 一致）。直接用 urllib/hf_hub_download 拉 resolve 端点会被 CDN 风控（403）。
"""

import argparse
import json
import os
import shutil

# 国内网络默认经 hf-mirror.com 镜像下载（用户已有 HF_ENDPOINT 时尊重原值）。
# 必须在 import transformers 之前设置：huggingface_hub 在 import 时缓存 endpoint。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 本地模型需拷贝的核心文件（存在的才拷贝）
_CORE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
)


def is_remote(model_path):
    """判断 model_path 是本地路径还是 HF hub 模型名。

    本地：已存在目录 / 以 . ~ 开头（相对路径）/ Windows 盘符绝对路径（E:\\ 或 E:/）。
    """
    if os.path.isdir(model_path):
        return False
    if model_path.startswith((".", "~")):
        return False
    if len(model_path) >= 3 and model_path[1] == ":" and model_path[2] in "/\\":
        return False  # Windows 盘符绝对路径
    return True


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def extract_chat_template(config):
    """从 tokenizer_config.json 提取默认 chat_template 字符串。

    兼容 transformers 的多种形态：
      - "..."                                    字符串
      - {"default": "...", "name": "..."}        多模板，取 default
      - {"default": {"system": "...", "default": "..."}}  嵌套形态（transformers 4.51）
    无模板返回 None。
    """
    if not isinstance(config, dict):
        return None
    ct = config.get("chat_template")
    while isinstance(ct, dict):
        if "default" in ct:
            nxt = ct["default"]
            if isinstance(nxt, str):
                return nxt
            ct = nxt
        else:
            for v in ct.values():
                if isinstance(v, str):
                    return v
            return None
    return ct if isinstance(ct, str) else None


def _export_via_transformers(model_path, out_dir):
    """远程模型：用 transformers 下载并保存 tokenizer 目录。

    与 tokenizer/zoo_test.py 一致——transformers 的下载链路（huggingface_hub）
    在用户网络下可正常走 hf-mirror；直接用 urllib 拉 resolve 端点会被 CDN 风控（403）。
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.save_pretrained(out_dir)


def _copy_local(model_path, out_dir):
    """本地模型目录：直接拷贝核心 tokenizer 文件。"""
    got = []
    for fname in _CORE_FILES:
        src = os.path.join(model_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            got.append(fname)
    return got


def export_tokenizer_hf(model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if is_remote(model_path):
        try:
            _export_via_transformers(model_path, out_dir)
        except Exception as e:
            raise RuntimeError("transformers 下载失败（{}）：{}".format(model_path, e))
    else:
        _copy_local(model_path, out_dir)

    # chat_template.jinja：从 tokenizer_config.json 提取默认模板写入
    cfg_path = os.path.join(out_dir, "tokenizer_config.json")
    config = _read_json(cfg_path) if os.path.isfile(cfg_path) else None
    tpl = extract_chat_template(config) if config else None
    if tpl:
        tpl_path = os.path.join(out_dir, "chat_template.jinja")
        with open(tpl_path, "w", encoding="utf-8") as f:
            f.write(tpl)

    if not (os.path.isfile(os.path.join(out_dir, "tokenizer.json")) or
            os.path.isfile(os.path.join(out_dir, "tokenizer.model"))):
        raise RuntimeError(
            "导出失败：{} 未找到 tokenizer.json / tokenizer.model".format(model_path))

    files = sorted(os.listdir(out_dir))
    print("[export_tokenizer_hf] {} 个文件 -> {}".format(len(files), out_dir))
    for f in files:
        print("  - {}".format(f))
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description="导出 HF tokenizer 目录供 RKNN3 新版 Tokenizer/ChatTemplate 加载")
    ap.add_argument("model_path", help="本地模型目录或 HF hub 模型名")
    ap.add_argument("output_dir", help="输出目录（存放 tokenizer.json / chat_template.jinja 等）")
    args = ap.parse_args()
    export_tokenizer_hf(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()

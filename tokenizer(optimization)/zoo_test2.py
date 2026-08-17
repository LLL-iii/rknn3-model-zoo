#!/usr/bin/env python3
"""Zoo model alignment test — always shows side-by-side C++ vs HF Python output."""

import os
import subprocess
import sys

# --- Config ---
TOKENIZE_DEMO = os.path.join(os.path.dirname(__file__),
    "install", "tokenizer_linux_x86", "demo", "tokenize_demo")

# {name: (hf_model_id, local_path_or_None, extra_kwargs_or_None)}
MODELS = {
    # ==================== BPE (ByteLevel) — 16 models ====================
    "Qwen3":           ("Qwen/Qwen3-1.7B",              None, None),
    "Qwen2_5":         ("Qwen/Qwen2.5-3B-Instruct",     None, None),
    "Qwen2_5_VL":      ("Qwen/Qwen2.5-VL-3B-Instruct",  None, None),
    "Qwen2_5_Omni":    ("Qwen/Qwen2.5-Omni-3B",         None, None),
    "Qwen3_VL":        ("Qwen/Qwen3-VL-4B-Instruct",     None, None),
    "Qwen3_VL_LoRA":   ("Qwen/Qwen3-VL-4B-Instruct",     None, {"_same_as": "Qwen3_VL"}),
    "Qwen3_ASR":       ("Qwen/Qwen3-ASR-0.6B",          None, None),
    "Qwen3_Embedding": ("Qwen/Qwen3-Embedding-0.6B",    None, None),
    "Qwen3_Reranker":  ("Qwen/Qwen3-Reranker-0.6B",     None, None),
    "Qwen3_TTS":       ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", None, None),
    "HY_MT_1_5":       ("Tencent-Hunyuan/HY-MT1.5-1.8B", None,
                        {"use_modelscope": True}),
    "Janus_Pro":       ("deepseek-ai/Janus-Pro-1B",      None,
                        {"use_saved_tok_for_comparison": True}),
    "SmolVLM":         ("HuggingFaceTB/SmolVLM-500M-Instruct", None, None),
    "SmolVLM2":        ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", None, None),
    "glm_edge":        ("THUDM/glm-edge-1.5b-chat",     None, None),
    "GME-Qwen2-VL":    ("Alibaba-NLP/GME-Qwen2-VL-2B-Instruct", None, None),

    # ==================== SPM (SentencePiece) — 4 models ====================
    "gemma4":          ("google/gemma-4-E2B-it",         None,
                        {"trust_remote_code": True, "use_fast": False,
                         "use_saved_tok_for_comparison": True}),
    "InternVLM":       ("OpenGVLab/InternVL3-1B",        None,
                        {"trust_remote_code": True}),
    "MiniCPM_V_4":     ("openbmb/MiniCPM-V-4",           None,
                        {"trust_remote_code": True}),
    "FastVLM":         ("llava-hf/llava-1.5-7b-hf",      None,
                        {"use_saved_tok_for_comparison": True}),
}

TEST_TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I've been working on AI for 3.5 years.",
    "你好，世界！",
    "人工智能是计算机科学的一个分支。",
    "a\nb\tc",
    "    多个空格    测试    ",
]


def load_tokenizer(model_id, extra_kwargs):
    kwargs = dict(extra_kwargs) if extra_kwargs else {}
    use_modelscope = kwargs.pop("use_modelscope", False)
    if use_modelscope:
        try:
            from modelscope import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except ImportError:
            print("  [WARN] modelscope not installed, trying HF mirror...")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_saved_tokenizer(save_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(save_dir, "tokenizer.json"))


def format_ids(ids, limit=40):
    """Format a list of int IDs, truncating with ellipsis if too long."""
    if len(ids) <= limit:
        return str(ids)
    return str(ids[:limit])[:-1] + f", ... ({len(ids)} total)]"


def parse_cpp_output(demo_output):
    """Parse token IDs from tokenize_demo stdout lines like '  450 -> \\' The\\''."""
    cpp_ids = []
    for line in demo_output:
        line = line.strip()
        if ' -> \'' in line and not line.startswith('Decode'):
            try:
                tok_id = int(line.split(' -> ')[0])
                cpp_ids.append(tok_id)
            except ValueError:
                pass
    return cpp_ids


def main():
    passed = 0
    failed = 0
    skipped = 0

    for name, (model_id, _, extra) in MODELS.items():
        print(f"\n{'='*70}")
        print(f"  {name}  ({model_id})")
        print(f"{'='*70}")

        try:
            hf = load_tokenizer(model_id, extra)
        except Exception as e:
            print(f"  [SKIP] Cannot load: {e}")
            skipped += 1
            continue

        save_dir = f"/tmp/rknn3_zoo_test_{name}"
        hf.save_pretrained(save_dir)

        kwargs = dict(extra) if extra else {}
        use_saved = kwargs.pop("use_saved_tok_for_comparison", False)
        py_tok = load_saved_tokenizer(save_dir) if use_saved else hf

        errors = 0
        for text in TEST_TEXTS:
            # --- Python side ---
            py_ids = py_tok.encode(text, add_special_tokens=False)
            if isinstance(py_ids, list):
                py_id_list = py_ids
            else:
                py_id_list = py_ids.ids

            # --- C++ side ---
            cpp_id_list = None
            cpp_error = None
            try:
                proc = subprocess.run(
                    [TOKENIZE_DEMO, "-t", save_dir, "-p", text, "--show-count", "--debug"],
                    capture_output=True, text=True, timeout=30)
                demo_output = proc.stdout.splitlines()
                demo_stderr = proc.stderr
            except Exception as e:
                cpp_error = str(e)
                demo_output = []
                demo_stderr = ""

            if cpp_error:
                cpp_id_list = None
            else:
                load_failed = any("load failed" in l or "Error: could not create"
                                  in l for l in demo_output)
                if load_failed:
                    cpp_id_list = None
                else:
                    cpp_id_list = parse_cpp_output(demo_output)

            # --- Render line ---
            text_label = repr(text)
            if len(text_label) > 55:
                text_label = text_label[:52] + "..."

            if cpp_id_list is not None and cpp_id_list == py_id_list:
                tag = "[OK]"
            else:
                tag = "[!!]"
                errors += 1

            print(f"  {tag} {text_label}")
            print(f"      HF : {format_ids(py_id_list)}")
            if cpp_id_list is not None:
                print(f"      C++: {format_ids(cpp_id_list)}")
            elif cpp_error:
                print(f"      C++: ERROR ({cpp_error})")
            else:
                print(f"      C++: LOAD FAILED")

        if errors == 0:
            print(f"  => PASS ({len(TEST_TEXTS)}/{len(TEST_TEXTS)})")
            passed += 1
        else:
            print(f"  => FAIL ({errors}/{len(TEST_TEXTS)} mismatches)")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Summary: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    print(f"Total models tested: {passed + failed + skipped}")
    print(f"{'='*70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

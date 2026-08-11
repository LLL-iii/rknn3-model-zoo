#!/usr/bin/env python3
"""Zoo model alignment test — compares C++ tokenizer output against HF Python."""

import os
import subprocess
import sys

# --- Config ---
TOKENIZE_DEMO = os.path.join(os.path.dirname(__file__),
    "build", "build_tokenizer_linux_x86_Release", "demo", "tokenize_demo")

# {name: (hf_model_id, local_path_or_None, extra_kwargs_or_None)}
MODELS = {
    # ==================== BPE (ByteLevel) — 16 models ====================
    "Qwen3":           ("Qwen/Qwen3-1.7B",              None, None),
#    "Qwen2_5":         ("Qwen/Qwen2.5-3B-Instruct",     None, None),
#    "Qwen2_5_VL":      ("Qwen/Qwen2.5-VL-3B-Instruct",  None, None),
#    "Qwen2_5_Omni":    ("Qwen/Qwen2.5-Omni-3B",         None, None),
#    "Qwen3_VL":        ("Qwen/Qwen3-VL-4B-Instruct",     None, None),
#    "Qwen3_VL_LoRA":   ("Qwen/Qwen3-VL-4B-Instruct",     None, {"_same_as": "Qwen3_VL"}),
#    "Qwen3_ASR":       ("Qwen/Qwen3-ASR-0.6B",          None, None),
#    "Qwen3_Embedding": ("Qwen/Qwen3-Embedding-0.6B",    None, None),
#    "Qwen3_Reranker":  ("Qwen/Qwen3-Reranker-0.6B",     None, None),
#    "Qwen3_TTS":       ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", None, None),
#    "HY_MT_1_5":       ("Tencent-Hunyuan/HY-MT1.5-1.8B", None,  # known: post_processor SpecialToken string ID
#                        {"use_modelscope": True}),
#    "Janus_Pro":       ("deepseek-ai/Janus-Pro-1B",      None,
#                       # save_pretrained rewrites Sequence→Metaspace
#                       {"use_saved_tok_for_comparison": True}),
#    "SmolVLM":         ("HuggingFaceTB/SmolVLM-500M-Instruct", None, None),
#    "SmolVLM2":        ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", None, None),
#    "glm_edge":        ("THUDM/glm-edge-1.5b-chat",     None, None),
#    "GME-Qwen2-VL":    ("Alibaba-NLP/GME-Qwen2-VL-2B-Instruct", None, None),

    # ==================== SPM (SentencePiece) — 5 models ====================
#    "gemma4":          ("google/gemma-4-E2B-it",         None,
#                        {"trust_remote_code": True, "use_fast": False,
#                         "use_saved_tok_for_comparison": True}),
#    "InternVLM":       ("OpenGVLab/InternVL3-1B",        None,
#                        {"trust_remote_code": True}),
#    "MiniCPM_V_4":     ("openbmb/MiniCPM-V-4",           None,
#                        {"trust_remote_code": True}),
#    "FastVLM":         ("llava-hf/llava-1.5-7b-hf",      None, None),
}

TEST_TEXTS = [
    "é ê ë è à â ä ô ö ù ü î ï ÿ ç ñ ¡ ¿ œ Œ € “ ”",
    
]

def load_tokenizer(model_id, extra_kwargs):
    """Load tokenizer, with optional ModelScope fallback for private models."""
    kwargs = dict(extra_kwargs) if extra_kwargs else {}

    use_modelscope = kwargs.pop("use_modelscope", False)
    if use_modelscope:
        try:
            from modelscope import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except ImportError:
            print("  [WARN] modelscope not installed, trying HF mirror...")
            pass

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_saved_tokenizer(save_dir):
    """Re-load the save_pretrained tokenizer so both Python and C++ use the
    same pretokenizer config. Some models rewrite the pretokenizer during
    save_pretrained (e.g. Sequence→Metaspace), which causes a mismatch if
    Python still uses the original in-memory tokenizer.

    Prefer the raw tokenizers.Tokenizer because it uses the exact
    tokenizer.json content with no Python-wrapper preprocessing."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(save_dir, "tokenizer.json"))

def main():
    passed = 0
    failed = 0
    skipped = 0

    for name, (model_id, _, extra) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}  ({model_id})")
        print(f"{'='*60}")

        try:
            hf = load_tokenizer(model_id, extra)
        except Exception as e:
            print(f"  [SKIP] Cannot load: {e}")
            skipped += 1
            continue

        save_dir = f"/tmp/rknn3_zoo_test_{name}"
        hf.save_pretrained(save_dir)

        # Some models change pretokenizer config during save_pretrained
        # (e.g. Sequence → Metaspace). For those, compare against the
        # saved tokenizer so both Python and C++ use the same config.
        kwargs = dict(extra) if extra else {}
        use_saved = kwargs.pop("use_saved_tok_for_comparison", False)
        py_tok = load_saved_tokenizer(save_dir) if use_saved else hf

        errors = 0
        for text in TEST_TEXTS:
            py_ids = py_tok.encode(text, add_special_tokens=False)
            if isinstance(py_ids, list):
                hf_ids = py_ids
            else:
                hf_ids = py_ids.ids

            try:
                proc = subprocess.run(
                    [TOKENIZE_DEMO, "-t", save_dir, "-p", text, "--show-count"],
                    capture_output=True, text=True, timeout=30)
                demo_output = proc.stdout.splitlines()
                demo_stderr = proc.stderr
            except FileNotFoundError:
                print(f"  [FATAL] tokenize_demo not found at {TOKENIZE_DEMO}")
                return 1
            except Exception as e:
                print(f"  [WARN] demo crashed on '{text[:30]}...': {e}")
                errors += 1
                continue

            # Check for load failure first
            load_failed = any("load failed" in l or "Error: could not create"
                              in l for l in demo_output)
            if load_failed:
                print(f"  [WARN] load failed for '{text[:30]}...'")
                stderr_lines = [l for l in demo_stderr.splitlines()
                                if "error" in l.lower() or "fail" in l.lower()
                                or "Error" in l or "E " in l[:4]]
                for l in stderr_lines[:5]:
                    print(f"    stderr: {l}")
                errors += 1
                continue

            # Parse "  450 -> ' The'" style lines
            cpp_ids = []
            for line in demo_output:
                line = line.strip()
                if ' -> \'' in line and not line.startswith('Decode'):
                    try:
                        tok_id = int(line.split(' -> ')[0])
                        cpp_ids.append(tok_id)
                    except ValueError:
                        pass

            if len(cpp_ids) == 0:
                if len(hf_ids) == 0:
                    # Both sides agree on no tokens (e.g. CJK text via
                    # save_pretrained Metaspace where all chars are OOV).
                    pass
                else:
                    print(f"  [WARN] no tokens parsed for '{text[:30]}...'")
                    print(f"  demo stdout: {demo_output[:5]}")
                    errors += 1
                continue

            if cpp_ids != hf_ids:
                print(f"  MISMATCH: '{text[:40]}...'")
                print(f"    HF : {hf_ids[:15]}...")
                print(f"    C++: {cpp_ids[:15]}...")
                errors += 1

        if errors == 0:
            print(f"  [PASS] All {len(TEST_TEXTS)} texts match")
            passed += 1
        else:
            print(f"  [FAIL] {errors} mismatch(es)")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} PASS, {failed} FAIL, {skipped} SKIP")
    print(f"Total models tested: {passed + failed + skipped}")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

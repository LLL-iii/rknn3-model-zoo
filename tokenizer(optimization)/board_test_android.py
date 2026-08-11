#!/usr/bin/env python3
"""RK3576 Android accuracy & latency test: C++ (NDK) vs Python on PC.

1. PC downloads EN+ZH datasets
2. PC pushes datasets + Android binary to RK3576 via ADB
3. For each model: board runs C++ batch encode, PC runs Python encode
4. Compare results, measure latency + memory, write MD report
"""

import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

HF_MIRROR = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_MIRROR

# Android board paths
ANDROID_DEMO   = "/data/local/tmp/tokenizer_android/demo/tokenize_demo"
ANDROID_MODELS = "/data/local/tmp/models"
ANDROID_DATA   = "/data/local/tmp/tokenizer_test_data"

PC_DEMO = None  # set below
PC_MODELS = "/tmp"
RESULT_DIR = "zoo_test_result"
SAMPLE_COUNT = 20000

random.seed(42)

MODELS = {
    "Qwen3":          ("Qwen/Qwen3-1.7B",              None, {"use_saved_tok": True}),
    "Qwen2_5":        ("Qwen/Qwen2.5-3B-Instruct",     None, {"use_saved_tok": True}),
    "Qwen2_5_VL":     ("Qwen/Qwen2.5-VL-3B-Instruct",  None, {"use_saved_tok": True}),
    "Qwen2_5_Omni":   ("Qwen/Qwen2.5-Omni-3B",         None, {"use_saved_tok": True}),
    "Qwen3_VL":       ("Qwen/Qwen3-VL-4B-Instruct",     None, {"use_saved_tok": True}),
    "Qwen3_ASR":      ("Qwen/Qwen3-ASR-0.6B",          None, {"use_saved_tok": True}),
    "Qwen3_Embedding":("Qwen/Qwen3-Embedding-0.6B",    None, {"use_saved_tok": True}),
    "Qwen3_Reranker": ("Qwen/Qwen3-Reranker-0.6B",     None, {"use_saved_tok": True}),
    "Qwen3_TTS":      ("Qwen/Qwen3-TTS-12Hz-1.7B-Base",None, {"use_saved_tok": True}),
    "HY_MT_1_5":      ("Tencent-Hunyuan/HY-MT1.5-1.8B", None, {"use_modelscope": True}),
    "Janus_Pro":      ("deepseek-ai/Janus-Pro-1B",      None, {"use_saved_tok": True}),
    "SmolVLM":        ("HuggingFaceTB/SmolVLM-500M-Instruct",    None, {}),
    "SmolVLM2":       ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", None, {}),
    "glm_edge":       ("THUDM/glm-edge-1.5b-chat",     None, {}),
    "GME-Qwen2-VL":   ("Alibaba-NLP/GME-Qwen2-VL-2B-Instruct", None, {"use_saved_tok": True}),
    "InternVLM":      ("OpenGVLab/InternVL3-1B",        None, {"trust_remote_code": True, "use_saved_tok": True}),
    "MiniCPM_V_4":    ("openbmb/MiniCPM-V-4",           None, {"trust_remote_code": True, "use_saved_tok": True, "strip_spm": True}),
    "FastVLM":        ("llava-hf/llava-1.5-7b-hf",      None, {}),
    "gemma4":         ("google/gemma-4-E2B-it",         None, {"trust_remote_code": True, "use_fast": False, "use_saved_tok": True}),
}

# ── Dataset ──────────────────────────────────────────────────────────────

def download_texts_en(n):
    from huggingface_hub import hf_hub_download
    local = hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt",
                            repo_type="dataset")
    with open(local, encoding="utf-8") as f:
        lines = [l.strip() for l in f if 20 <= len(l.strip()) <= 4000]
    random.shuffle(lines)
    return lines[:n]


def download_texts_zh(n):
    from huggingface_hub import hf_hub_download
    repo, fname, col = "shibing624/alpaca-zh", "alpaca_gpt4_data_zh.json", "output"
    local = hf_hub_download(repo, fname, repo_type="dataset")
    data = json.load(open(local, encoding="utf-8"))
    texts = []
    for item in data:
        txt = str(item.get(col, "") if isinstance(item, dict) else item).strip()
        if 20 <= len(txt) <= 4000:
            texts.append(txt)
    random.shuffle(texts)
    return texts[:n]


# ── Tokenizer ────────────────────────────────────────────────────────────

def load_tokenizer(model_id, extra):
    kwargs = dict(extra) if extra else {}
    if kwargs.pop("use_modelscope", False):
        from modelscope import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, **kwargs)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_saved_tokenizer(save_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(save_dir, "tokenizer.json"))


def py_encode(tok, text):
    r = tok.encode(text, add_special_tokens=False)
    return list(r) if isinstance(r, list) else r.ids


# ── Board communication ─────────────────────────────────────────────────

def check_android():
    """Verify ADB connectivity and demo binary present."""
    try:
        out = subprocess.run(["adb", "shell", "echo OK"],
                             capture_output=True, text=True, timeout=10).stdout
        if "OK" not in out:
            print("ERROR: adb shell failed, check USB connection")
            return False
    except Exception as e:
        print(f"ERROR: adb unreachable ({e})")
        return False

    # Check demo exists (Android: busybox test -f is safer than test -x)
    out = subprocess.run(
        ["adb", "shell", f"ls {ANDROID_DEMO}"],
        capture_output=True, text=True, timeout=10).stdout
    if ANDROID_DEMO not in out:
        print(f"ERROR: demo not found at {ANDROID_DEMO}")
        return False

    return True


def android_batch_encode(model_path, texts, timeout=600):
    """Run batch encode on Android board, return (ids, elapsed_ms, peak_rss_kb)."""
    import shlex
    # Build batch input
    parts = [str(len(texts))]
    for t in texts:
        b = t.encode("utf-8")
        parts.append(f"{len(b)} {t}")
    stdin_str = "\n".join(parts)

    tmp_in  = f"{ANDROID_DATA}/in.txt"
    tmp_out = f"{ANDROID_DATA}/out.txt"
    tmp_rss = f"{ANDROID_DATA}/rss.txt"

    # Ensure data dir
    subprocess.run(["adb", "shell", f"mkdir -p {ANDROID_DATA}"],
                   capture_output=True, timeout=10)

    # Push input
    subprocess.run(
        ["adb", "shell", f"cat > {tmp_in}"],
        input=stdin_str, text=True, capture_output=True, timeout=30)

    demo_q = shlex.quote(ANDROID_DEMO)
    mp_q   = shlex.quote(model_path)

    # Shell wrapper: background demo, poll VmRSS in /proc/$pid/status
    wrapper = (
        f"rm -f {tmp_out} {tmp_rss}; "
        f"peak=0; "
        f"({demo_q} --stdin-batch -t {mp_q} < {tmp_in} > {tmp_out}) & "
        f"pid=$!; "
        "while [ -d /proc/$pid ] 2>/dev/null; do "
        "  rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null); "
        "  [ -n \"$rss\" ] && [ \"$rss\" -gt \"$peak\" ] && peak=$rss; "
        "  sleep 0.1; "
        "done; "
        f"wait $pid; "
        f"echo $peak > {tmp_rss}"
    )

    t0 = time.perf_counter()
    subprocess.run(["adb", "shell", wrapper],
                   capture_output=True, text=True, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Read peak RSS
    peak_rss_kb = 0
    try:
        rss_out = subprocess.run(
            ["adb", "shell", f"cat {tmp_rss}"],
            capture_output=True, text=True, timeout=5).stdout
        peak_rss_kb = int(rss_out.strip() or "0")
    except (ValueError, Exception):
        pass

    # Read output
    stdout = subprocess.run(
        ["adb", "shell", f"cat {tmp_out}"],
        capture_output=True, text=True, timeout=30).stdout

    # Parse
    results = []
    for line in stdout.splitlines():
        toks = line.split()
        if not toks or toks[0] == "ERR":
            results.append(None)
        elif toks[0] == "0":
            results.append([])
        else:
            try:
                n = int(toks[0])
                results.append([int(x) for x in toks[1:n+1]])
            except (ValueError, IndexError):
                results.append(None)

    return results, elapsed_ms, peak_rss_kb


def compute_f1(py_ids, cpp_ids):
    if cpp_ids is None:
        return 0.0
    if len(py_ids) == 0 and len(cpp_ids) == 0:
        return 1.0
    if len(py_ids) == 0 or len(cpp_ids) == 0:
        return 0.0
    from collections import Counter
    pc = Counter(py_ids)
    cc = Counter(cpp_ids)
    common = sum((pc & cc).values())
    p = common / len(cpp_ids) if cpp_ids else 0.0
    r = common / len(py_ids) if py_ids else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 70)
    print("  ANDROID BOARD TEST — RK3576 Dataset Accuracy & Latency")
    print("=" * 70)
    print(f"  Demo:  {ANDROID_DEMO}")
    print(f"  Data:  {ANDROID_DATA}")

    # Check board
    print("\n[CHECK] Board connectivity...")
    if not check_android():
        return 1
    print("  Board reachable, demo found.")

    # Download datasets
    print(f"\n[DS] Downloading ({HF_MIRROR}) ...")
    try:
        en_texts = download_texts_en(SAMPLE_COUNT)
        zh_texts = download_texts_zh(SAMPLE_COUNT)
        print(f"  EN: {len(en_texts):,} texts, ZH: {len(zh_texts):,} texts")
    except Exception as e:
        print(f"  Download failed: {e}"); return 1

    all_texts = en_texts + zh_texts
    total_texts = len(all_texts)

    # Discover board models
    out = subprocess.run(
        ["adb", "shell", f"ls {ANDROID_MODELS}/"],
        capture_output=True, text=True, timeout=10).stdout
    board_models = set(out.split())
    print(f"\n[MODELS] Board has {len(board_models)} dirs")

    report = []

    for idx, (name, (mid, _, extra)) in enumerate(MODELS.items()):
        model_dir_name = f"rknn3_zoo_test_{name}"
        if model_dir_name not in board_models:
            print(f"\n[{idx+1:>2}/{len(MODELS)}] {name} — SKIP (not on board)")
            continue

        kw = dict(extra) if extra else {}
        strip_spm = kw.pop("strip_spm", False)
        use_saved = kw.pop("use_saved_tok", False)

        board_path = f"{ANDROID_MODELS}/{model_dir_name}"
        pc_path = f"{PC_MODELS}/{model_dir_name}"

        print(f"\n[{idx+1:>2}/{len(MODELS)}] {name}  ({mid})")

        # Python tokenizer
        try:
            if use_saved and os.path.exists(os.path.join(pc_path, "tokenizer.json")):
                py = load_saved_tokenizer(pc_path)
                if strip_spm:
                    spm = os.path.join(pc_path, "tokenizer.model")
                    if os.path.exists(spm): os.remove(spm)
            else:
                hf = load_tokenizer(mid, extra)
                hf.save_pretrained(pc_path)
                if strip_spm:
                    spm = os.path.join(pc_path, "tokenizer.model")
                    if os.path.exists(spm): os.remove(spm)
                py = load_saved_tokenizer(pc_path) if use_saved else hf
        except Exception as e:
            print(f"  [SKIP] Load failed: {e}")
            continue

        # C++ encode on Android
        print(f"  C++ encode ({total_texts} texts)...")
        try:
            cpp_ids_list, encode_ms, peak_rss_kb = android_batch_encode(
                board_path, all_texts, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] >10 min")
            continue
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        # Python encode
        print(f"  Python encode...")
        t0 = time.perf_counter()
        py_ids_list = [py_encode(py, t) for t in all_texts]
        py_encode_ms = (time.perf_counter() - t0) * 1000

        # Compare
        ok, fail = 0, 0
        sum_f1 = 0.0
        mismatches = []

        for i in range(total_texts):
            pids = py_ids_list[i]
            cids = cpp_ids_list[i] if i < len(cpp_ids_list) else None
            f1 = compute_f1(pids, cids)
            sum_f1 += f1
            if cids is None:
                fail += 1
            elif cids == pids:
                ok += 1
            else:
                fail += 1
                if len(mismatches) < 20:
                    mismatches.append({
                        "lang": "en" if i < len(en_texts) else "zh",
                        "text": all_texts[i][:200],
                        "py_len": len(pids),
                        "cpp_len": len(cids),
                        "f1": round(f1, 4),
                    })

        avg_f1 = sum_f1 / total_texts if total_texts else 0
        acc = ok / (ok + fail) * 100 if (ok + fail) else 0
        enc_per_text = encode_ms / total_texts if total_texts else 0
        py_per_text = py_encode_ms / total_texts if total_texts else 0
        speedup = py_per_text / enc_per_text if enc_per_text else 0

        print(f"  Acc: {acc:.2f}%  F1: {avg_f1:.4f}  "
              f"C++: {enc_per_text:.2f}ms/t  Py: {py_per_text:.2f}ms/t  "
              f"{speedup:.1f}x  RSS: {peak_rss_kb/1024:.1f} MB")

        report.append({
            "model": name,
            "accuracy_pct": round(acc, 2),
            "token_f1": round(avg_f1, 4),
            "cpp_encode_total_ms": round(encode_ms, 1),
            "py_encode_total_ms": round(py_encode_ms, 1),
            "cpp_per_text_ms": round(enc_per_text, 2),
            "py_per_text_ms": round(py_per_text, 2),
            "speedup": round(speedup, 1),
            "peak_rss_kb": peak_rss_kb,
            "mismatches": fail,
            "samples": mismatches[:5],
        })

    # ── Report ──
    print(f"\n{'=' * 70}")
    print(f"  ANDROID RK3576 RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Model':<18} {'Acc':>7} {'F1':>7} {'C++(ms)':>9} {'Py(ms)':>9} {'x':>5} {'RSS(MB)':>8}")
    print("-" * 76)
    for r in report:
        rss_mb = r.get('peak_rss_kb', 0) / 1024
        print(f"{r['model']:<18} {r['accuracy_pct']:>6.2f}% {r['token_f1']:>6.4f} "
              f"{r['cpp_per_text_ms']:>8.2f} {r['py_per_text_ms']:>8.2f} "
              f"{r['speedup']:>4.1f}x {rss_mb:>7.1f}")

    ok_acc = sum(1 for r in report if r['accuracy_pct'] >= 99.99)
    print(f"\n100% models: {ok_acc}/{len(report)}")

    # ── MD report ──
    now = time.strftime("%Y-%m-%d %H:%M")
    md_lines = [
        "# Android Board Test Report (RK3576)",
        f"**Date**: {now}",
        f"**Chip**: RK3576",
        f"**Platform**: Android (NDK r23c, Clang 12 / C++17, arm64-v8a)",
        f"**Demo**: `{ANDROID_DEMO}` (static, stripped)",
        "",
        "## Datasets",
        f"| Language | Dataset | Texts | Chars |",
        f"|:---------|:--------|------:|------:|",
        f"| EN | roneneldan/TinyStories | {len(en_texts):,} | {sum(len(t) for t in en_texts):,} |",
        f"| ZH | shibing624/alpaca-zh | {len(zh_texts):,} | {sum(len(t) for t in zh_texts):,} |",
        f"| **Total** | | **{total_texts:,}** | **{sum(len(t) for t in all_texts):,}** |",
        "",
        "---",
        "",
        "## Results",
        "",
        f"| Model | Acc | Token F1 | C++ (ms/text) | Python (ms/text) | Speedup | Peak RSS (MB) |",
        f"|:------|----:|:--------:|-------------:|-----------------:|-------:|-------------:|",
    ]
    for r in report:
        rss_mb = r.get('peak_rss_kb', 0) / 1024
        md_lines.append(
            f"| {r['model']} | {r['accuracy_pct']:.2f}% | {r['token_f1']:.4f} | "
            f"{r['cpp_per_text_ms']:.2f} | {r['py_per_text_ms']:.2f} | "
            f"{r['speedup']:.1f}x | {rss_mb:.1f} |")

    md_lines += [
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Models tested**: {len(report)}",
        f"- **100% accuracy**: {ok_acc}/{len(report)}",
    ]
    if report:
        avg_acc = sum(r['accuracy_pct'] for r in report) / len(report)
        avg_f1 = sum(r['token_f1'] for r in report) / len(report)
        avg_rss = sum(r.get('peak_rss_kb', 0) for r in report) / len(report) / 1024
        md_lines += [
            f"- **Avg accuracy**: {avg_acc:.2f}%",
            f"- **Avg Token F1**: {avg_f1:.4f}",
            f"- **Avg Peak RSS**: {avg_rss:.1f} MB",
        ]

    # Mismatch JSON (not in MD)
    mm_detail = {r['model']: r.get('samples', []) for r in report if r['mismatches'] > 0}
    if mm_detail:
        with open(os.path.join(RESULT_DIR, "board_test_mismatch_android.json"), "w", encoding="utf-8") as f:
            json.dump(mm_detail, f, indent=2, ensure_ascii=False)

    rpt_path = os.path.join(RESULT_DIR, "board_test_report_android.md")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Report saved: {rpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

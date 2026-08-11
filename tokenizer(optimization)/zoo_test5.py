#!/usr/bin/env python3
"""Round-trip fidelity benchmark — C++ encode → decode back to text.

Tests whether the C++ tokenizer is fully lossless: encode a text to
token IDs, then decode those IDs back to text, check byte-identical
match with the original input.

Uses TinyStories (EN) + Alpaca-ZH (ZH) datasets — 10,000 texts each.
Writes: zoo_test_result/RT_REPORT.md + token_rt_details.json
"""

import io
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict, defaultdict

HF_MIRROR = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_MIRROR

TOKENIZE_DEMO = os.path.join(os.path.dirname(__file__),
    "build", "build_tokenizer_linux_x86_64_Release", "demo", "tokenize_demo")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "zoo_test_result")
CKPT_FILE  = os.path.join(os.path.dirname(__file__), ".zoo_test5_ckpt.json")

random.seed(42)

MODELS = {
    "Qwen3":          ("Qwen/Qwen3-1.7B",              None, {"use_saved_tok": True}),
    "Qwen2_5":        ("Qwen/Qwen2.5-3B-Instruct",     None, {"use_saved_tok": True}),
    "Qwen2_5_VL":     ("Qwen/Qwen2.5-VL-3B-Instruct",  None, {"use_saved_tok": True}),
    "Qwen2_5_Omni":   ("Qwen/Qwen2.5-Omni-3B",         None, {"use_saved_tok": True}),
    "Qwen3_VL":       ("Qwen/Qwen3-VL-4B-Instruct",     None, {"use_saved_tok": True}),
    "Qwen3_VL_LoRA":  ("Qwen/Qwen3-VL-4B-Instruct",     None, {"_same_as": "Qwen3_VL"}),
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
    "gemma4":         ("google/gemma-4-E2B-it",         None,
                       {"trust_remote_code": True, "use_fast": False, "use_saved_tok": True}),
    "InternVLM":      ("OpenGVLab/InternVL3-1B",        None,
                       {"trust_remote_code": True, "use_saved_tok": True}),
    "MiniCPM_V_4":    ("openbmb/MiniCPM-V-4",           None,
                       {"trust_remote_code": True, "use_saved_tok": True}),
    "FastVLM":        ("llava-hf/llava-1.5-7b-hf",      None, {}),
}

DATASET_META = {
    "en": {"name": "roneneldan/TinyStories", "description": "English children's stories (GPT-3.5/4 generated)"},
    "zh": {"name": "shibing624/alpaca-zh",  "description": "Chinese instruction-following dataset (Alpaca format)"},
}

# ── Dataset downloads ────────────────────────────────────────────────────

def download_texts_en(n):
    from huggingface_hub import hf_hub_download
    local = hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt",
                            repo_type="dataset")
    with open(local, encoding="utf-8") as f:
        lines = [l.strip() for l in f if 20 <= len(l.strip()) <= 4000]
    if len(lines) < n:
        lines = lines * (n // len(lines) + 1)
    random.shuffle(lines)
    return lines[:n]


def download_texts_zh(n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for repo, fname, col in [
        ("shibing624/alpaca-zh",          "alpaca_gpt4_data_zh.json",     "output"),
        ("BelleGroup/train_0.5M_CN",      "Belle_open_source_0.5M.json", "instruction"),
        ("BelleGroup/train_1M_CN",        "Belle_open_source_1M.json",   "instruction"),
        ("BelleGroup/train_2M_CN",        "Belle_open_source_2M.json",   "instruction"),
    ]:
        try:
            local = hf_hub_download(repo, fname, repo_type="dataset")
            data = json.load(open(local, encoding="utf-8"))
            texts = []
            for item in data:
                txt = str(item.get(col, "") if isinstance(item, dict) else item).strip()
                if 20 <= len(txt) <= 4000:
                    texts.append(txt)
            texts = list(set(texts))
            if len(texts) < n:
                texts = texts * (n // len(texts) + 1)
            random.shuffle(texts)
            if len(texts) >= n:
                DATASET_META["zh"] = {"name": repo, "description": "Chinese instruction-following dataset"}
                return texts[:n]
        except Exception:
            continue
    raise RuntimeError("No Chinese dataset available")

# ── Helpers ───────────────────────────────────────────────────────────────

def load_tokenizer(model_id, extra):
    kwargs = dict(extra) if extra else {}
    if kwargs.pop("use_modelscope", False):
        try:
            from modelscope import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except ImportError:
            pass
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def char_f1(original, decoded):
    """Token-level character F1 between original and decoded text."""
    if not original and not decoded:
        return 1.0, 1.0, 1.0
    if not original or not decoded:
        return 0.0, 0.0, 0.0
    oc = Counter(original)
    dc = Counter(decoded)
    common = sum((oc & dc).values())
    p = common / len(decoded) if decoded else 0.0
    r = common / len(original) if original else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def run_cpp_roundtrip(save_dir, texts):
    """Two-step C++ round-trip: encode batch → decode batch.
    Returns (decoded_strings, elapsed, peak_kb)."""

    # Step 1: encode batch → token IDs
    parts = [str(len(texts)).encode("utf-8")]
    for t in texts:
        b = t.encode("utf-8")
        parts.append(str(len(b)).encode("utf-8") + b" " + b)
    stdin_bytes = b"\n".join(parts)

    time_args = ["/usr/bin/time", "-v", TOKENIZE_DEMO, "--stdin-batch", "-t", save_dir]

    t0 = time.perf_counter()
    proc = subprocess.run(time_args, input=stdin_bytes, capture_output=True, timeout=1200)
    enc_elapsed = time.perf_counter() - t0

    peak_kb = 0
    for line in proc.stderr.decode("utf-8", errors="replace").splitlines():
        if "Maximum resident set size" in line:
            peak_kb = int(line.split(":")[1].strip())
            break

    # Parse token ID lists from encode output
    id_lines = []   # list of "id1 id2 ... idN" strings
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("ERR"):
            id_lines.append("")
        elif line == "0":
            id_lines.append("")
        else:
            toks = line.split()
            try:
                n = int(toks[0])
                id_lines.append(" ".join(toks[1:n+1]))
            except (ValueError, IndexError):
                id_lines.append("")
    while len(id_lines) < len(texts):
        id_lines.append("")

    # Step 2: decode batch → text
    dec_parts = [str(len(id_lines)).encode("utf-8")]
    for ids in id_lines:
        dec_parts.append(ids.encode("utf-8"))
    dec_bytes = b"\n".join(dec_parts)

    t1 = time.perf_counter()
    dec_proc = subprocess.run(
        [TOKENIZE_DEMO, "--stdin-decode", "-t", save_dir],
        input=dec_bytes, capture_output=True, timeout=1200)
    dec_elapsed = time.perf_counter() - t1

    # Parse length-prefixed decode output at BYTE level:
    # "BYTELEN BYTES\n" where BYTES can contain embedded \n.
    decoded = []
    data = dec_proc.stdout
    offset = 0
    while offset < len(data):
        # Find the space separator
        sp = data.find(b' ', offset)
        if sp == -1:
            break
        try:
            blen = int(data[offset:sp])
        except ValueError:
            offset = sp + 1
            continue
        payload_start = sp + 1
        payload_end   = payload_start + blen
        if payload_end > len(data):
            # Truncated — take what we have
            decoded_text = data[payload_start:].decode("utf-8", errors="replace")
            decoded.append(decoded_text)
            break
        decoded_text = data[payload_start:payload_end].decode("utf-8", errors="replace")
        decoded.append(decoded_text)
        # Skip past payload + trailing newline
        offset = payload_end + 1  # +1 for \n

    while len(decoded) < len(texts):
        decoded.append(None)

    total_elapsed = enc_elapsed + dec_elapsed
    return decoded, total_elapsed, peak_kb

# ── Report ────────────────────────────────────────────────────────────────

def write_report(rt_list, lat_list, langs, en_n, zh_n, errors_by_model):
    path = os.path.join(RESULT_DIR, "RT_REPORT.md")

    lines = [
        "# Tokenizer Round-Trip Fidelity Report",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Models tested**: {len(rt_list)}",
        "",
        "## Method",
        "1. Encode raw text → token IDs via C++ tokenizer.",
        "2. Decode those IDs back → text via C++ tokenizer (`Decode()`).",
        "3. Compare decoded text with original byte-by-byte (exact match) and character-level F1.",
        "",
        "## Datasets",
        f"| Language | Dataset | Description | Texts |",
        f"|:---------|:--------|:------------|------:|",
        f"| EN | `{DATASET_META['en']['name']}` | {DATASET_META['en']['description']} | {en_n} |",
        f"| ZH | `{DATASET_META['zh']['name']}` | {DATASET_META['zh']['description']} | {zh_n} |",
        f"**Source mirror**: {HF_MIRROR}",
        "",
        "## 1. Round-Trip per Model × Language",
        "",
    ]

    hdr = "| Model | " + " | ".join(f"{l}" for l in langs) + " | Overall Exact | Char F1 | C++ (ms) | Peak RSS (MB) |"
    sep  = "|:------|" + ":----:|" * len(langs) + ":-------------:|:------:|--------:|-------------:|"
    lines.append(hdr); lines.append(sep)

    for a, l2 in zip(rt_list, lat_list):
        if a.get("status") == "SKIP":
            lines.append(f"| {a['model']:<12} | SKIP |")
            continue
        row = f"| {a['model']:<12} |"
        for l in langs:
            e = a.get("per_language", {}).get(l, {"ok": 0, "total": 0})
            p = e["ok"] / e["total"] * 100 if e["total"] > 0 else 0
            row += f" {p:5.1f}% |"
        row += f" {a['exact_match_pct']:>5.1f}% | {a['avg_char_f1']:.4f} | {l2['cpp_avg_latency_ms']:>7.3f} | {l2['cpp_peak_rss_kb']/1024:>11.1f} |"
        lines.append(row)

    n100 = sum(1 for a in rt_list if a.get("exact_match_pct", 0) >= 99.99)
    avg_em = sum(a.get("exact_match_pct", 0) for a in rt_list if a.get("status") != "SKIP") / max(len(rt_list), 1)
    avg_f1 = sum(a.get("avg_char_f1", 0) for a in rt_list if a.get("status") != "SKIP") / max(len(rt_list), 1)

    lines.append("")
    lines.append("## 2. Summary\n")
    lines.append(f"- **100% exact round-trip models**: {n100}/{len(rt_list)}")
    lines.append(f"- **Avg exact match**: {avg_em:.1f}%")
    lines.append(f"- **Avg char-level F1**: {avg_f1:.4f}")
    if any(len(v) > 0 for v in errors_by_model.values()):
        lines.append(f"- **Round-trip errors**: saved to `token_rt_errors.json` (not shown in report)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✓ Report: {path}")

# ── Checkpoint ───────────────────────────────────────────────────────────

def load_ckpt():
    if os.path.exists(CKPT_FILE):
        try:
            data = json.load(open(CKPT_FILE, encoding="utf-8"))
            return set(data.get("done", [])), data.get("rt", []), data.get("lat", []), data.get("err", {})
        except Exception:
            pass
    return set(), [], [], {}

def save_ckpt(done_names, rt_list, lat_list, errors):
    tmp = CKPT_FILE + ".tmp"
    ser = {"done": sorted(done_names), "rt": rt_list, "lat": lat_list,
           "err": {k: v[:50] for k, v in errors.items()}}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ser, f, ensure_ascii=False)
    os.replace(tmp, CKPT_FILE)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    done_set, rt_rpt, lat_rpt, err_by_model = load_ckpt()

    print("=" * 70)
    print("  ZOO TEST 5 — Round-Trip Fidelity Benchmark")
    print("=" * 70)
    if done_set:
        print(f"  [CKPT] {len(done_set)} model(s) already done, resuming ...")

    N_EN = 10000
    N_ZH = 10000

    print(f"\n[DS] Downloading from {HF_MIRROR} ...")
    try:
        en_texts = download_texts_en(N_EN)
        print(f"  ✓ EN: {len(en_texts)} texts")
    except Exception as e:
        print(f"  ✗ EN: {e}"); return 1

    try:
        zh_texts = download_texts_zh(N_ZH)
        print(f"  ✓ ZH: {len(zh_texts)} texts")
    except Exception as e:
        print(f"  ✗ ZH: {e}"); return 1

    print(f"  EN {sum(len(t) for t in en_texts):,} chars  |  ZH {sum(len(t) for t in zh_texts):,} chars\n")

    langs = ["en (TinyStories)", "zh (Alpaca-ZH)"]
    texts_by_lang = {langs[0]: en_texts, langs[1]: zh_texts}
    all_texts = en_texts + zh_texts
    total_all = len(all_texts)

    same_as = {}

    try:
        for idx_all, (name, (mid, _, extra)) in enumerate(MODELS.items()):
            if name in done_set:
                print(f"\n[{idx_all+1:>2}/{len(MODELS)}] {name} — skipped")
                continue
            kw = dict(extra) if extra else {}
            share = kw.pop("_same_as", None)
            if share:
                same_as[name] = share; continue

            print(f"[{idx_all+1:>2}/{len(MODELS)}] {name}  ({mid})")
            try:
                hf = load_tokenizer(mid, extra)
            except Exception as e:
                print(f"   [SKIP] {e}")
                rt_rpt.append({"model": name, "status": "SKIP"})
                lat_rpt.append({"model": name, "status": "SKIP"})
                done_set.add(name); save_ckpt(done_set, rt_rpt, lat_rpt, err_by_model)
                continue

            sd = f"/tmp/rknn3_zoo_test_{name}"
            hf.save_pretrained(sd)
            if kw.pop("strip_spm", False):
                spm = os.path.join(sd, "tokenizer.model")
                if os.path.exists(spm): os.remove(spm)

            decoded_texts, batch_time, peak_kb = run_cpp_roundtrip(sd, all_texts)
            if batch_time <= 0: batch_time = 0.001

            exact_ok = 0
            sum_f1 = 0.0
            pl = defaultdict(lambda: {"ok": 0, "total": 0})
            errors = []

            done = 0
            for label, lt in texts_by_lang.items():
                for i in range(len(lt)):
                    orig = all_texts[done]
                    dec  = decoded_texts[done] if done < len(decoded_texts) else None
                    pl[label]["total"] += 1

                    if dec is not None and orig == dec:
                        exact_ok += 1; pl[label]["ok"] += 1
                    p, r, f = char_f1(orig, dec or "")
                    sum_f1 += f
                    if dec is None or orig != dec:
                        if len(errors) < 50:
                            errors.append({"lang": label, "original": orig,
                                           "decoded": dec or "(null)",
                                           "char_f1": round(f, 4)})

                    done += 1
                    if done % 500 == 0 or done == total_all:
                        pct = done / total_all * 100
                        bar = "#" * int(pct/2) + "-" * (50 - int(pct/2))
                        print(f"\r   [{bar}] {done}/{total_all} ({pct:.0f}%)", end="")

            print()
            tot = sum(v["total"] for v in pl.values())
            em_pct = exact_ok / tot * 100 if tot else 0
            avg_f1 = sum_f1 / tot if tot else 0
            cms = batch_time / tot * 1000 if tot else 0

            print(f"   Exact: {em_pct:.2f}% ({exact_ok}/{tot})  Char F1: {avg_f1:.4f}  "
                  f"C++ {cms:.2f}ms  RSS: {peak_kb/1024:.1f}MB")
            for l in langs:
                e = pl[l]; p = e["ok"] / e["total"] * 100 if e["total"] else 0
                m = "✓" if p >= 99.99 else ("⚠" if p >= 90 else "✗")
                print(f"     {m} {l:<24} {p:5.1f}% ({e['ok']}/{e['total']})")

            rt_rpt.append({
                "model": name, "model_id": mid, "status": "PASS",
                "exact_matches": exact_ok, "total_texts": tot,
                "exact_match_pct": round(em_pct, 2),
                "avg_char_f1": round(avg_f1, 4),
                "per_language": {k: {"ok": v["ok"], "total": v["total"]}
                                 for k, v in pl.items()},
            })
            lat_rpt.append({
                "model": name, "cpp_avg_latency_ms": round(cms, 4),
                "cpp_peak_rss_kb": peak_kb,
            })
            err_by_model[name] = errors

            done_set.add(name)
            save_ckpt(done_set, rt_rpt, lat_rpt, err_by_model)
            write_report(rt_rpt, lat_rpt, langs, N_EN, N_ZH, err_by_model)

    except KeyboardInterrupt:
        print(f"\n\n  ⚠ Interrupted — {len(done_set)} models saved")
        return 0

    for alias, target in same_as.items():
        for lst in [rt_rpt, lat_rpt]:
            for r in lst:
                if r.get("model") == target:
                    r2 = dict(r); r2["model"] = alias; lst.append(r2); break
        if target in err_by_model:
            err_by_model[alias] = err_by_model[target]
        done_set.add(alias)

    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)

    # Save error details (separate file, not in report)
    err_path = os.path.join(RESULT_DIR, "token_rt_errors.json")
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in err_by_model.items() if v}, f, indent=2, ensure_ascii=False)

    with open(os.path.join(RESULT_DIR, "token_rt_report.json"), "w", encoding="utf-8") as f:
        json.dump(rt_rpt, f, indent=2, ensure_ascii=False)
    write_report(rt_rpt, lat_rpt, langs, N_EN, N_ZH, err_by_model)

    ok = sum(a.get("exact_matches", 0) for a in rt_rpt if a.get("status") != "SKIP")
    print(f"\n{'='*70}\n  TOTAL exact matches: {ok:,} / {sum(a.get('total_texts',0) for a in rt_rpt if a.get('status')!='SKIP'):,}\n{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

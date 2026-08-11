#!/usr/bin/env python3
"""Dataset accuracy, latency & memory benchmark — 2000 real EN + ZH texts.

Downloads TinyStories (EN) and Alpaca-ZH (ZH) from hf-mirror.
Encodes all texts with both C++ demo and HF Python tokenizer.
Collects mismatched samples and C++ peak memory (via /usr/bin/time -v).

Writes: zoo_test_result/DATASET_REPORT.md
        zoo_test_result/token_acc_dataset_report.json
        zoo_test_result/token_latency_report.json
        zoo_test_result/token_mismatch_details.json
"""

import io
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

HF_MIRROR = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_MIRROR

TOKENIZE_DEMO = os.path.join(os.path.dirname(__file__),
    "build", "build_tokenizer_linux_x86_Release", "demo", "tokenize_demo")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "zoo_test_result")
CKPT_FILE  = os.path.join(os.path.dirname(__file__), ".zoo_test4_ckpt.json")

random.seed(42)

# ── Models ────────────────────────────────────────────────────────────────

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
#    "gemma4":         ("google/gemma-4-E2B-it",         None,
#                       {"trust_remote_code": True, "use_fast": False, "use_saved_tok": True}),
    "InternVLM":      ("OpenGVLab/InternVL3-1B",        None,
                       {"trust_remote_code": True, "use_saved_tok": True}),
    "MiniCPM_V_4":    ("openbmb/MiniCPM-V-4",           None,
                       {"trust_remote_code": True, "use_saved_tok": True, "strip_spm": True}),
    "FastVLM":        ("llava-hf/llava-1.5-7b-hf",      None, {}),
}

DATASET_META = {
    "en": {"name": "roneneldan/TinyStories",
           "description": "English children's stories (GPT-3.5/4 generated)"},
    "zh": {"name": "shibing624/alpaca-zh",
           "description": "Chinese instruction-following dataset (Alpaca format)"},
}

# ── Dataset downloads ────────────────────────────────────────────────────

def download_texts_en(n):
    from huggingface_hub import hf_hub_download
    local = hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt",
                            repo_type="dataset")
    # Pick more texts since we increased N (datasets have >>10k each)
    with open(local, encoding="utf-8") as f:
        lines = [l.strip() for l in f if 20 <= len(l.strip()) <= 4000]
    if len(lines) < n:
        lines = lines * (n // len(lines) + 1)  # repeat if dataset is small
    random.shuffle(lines)
    return lines[:n]


def download_texts_zh(n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    CANDIDATES = [
        ("shibing624/alpaca-zh",          "alpaca_gpt4_data_zh.json",     "output"),
        ("BelleGroup/train_0.5M_CN",      "Belle_open_source_0.5M.json", "instruction"),
        ("BelleGroup/train_1M_CN",        "Belle_open_source_1M.json",   "instruction"),
        ("BelleGroup/train_2M_CN",        "Belle_open_source_2M.json",   "instruction"),
    ]
    for repo, fname, col in CANDIDATES:
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
                texts = texts * (n // len(texts) + 1)  # expand pool
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


def load_saved_tokenizer(save_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(save_dir, "tokenizer.json"))


def get_py_ids(tok, text):
    r = tok.encode(text, add_special_tokens=False)
    return list(r) if isinstance(r, list) else r.ids


def compute_f1_score(py_ids, cpp_ids):
    """Token-level F1 score via intersection of the two ID sequences.
    Exactly matching sequences → 1.0.  One side empty → 0.0."""
    if cpp_ids is None:
        return 0.0, 0.0, 0.0
    if len(py_ids) == 0 and len(cpp_ids) == 0:
        return 1.0, 1.0, 1.0
    # Weighted Jaccard: intersection of multiset token IDs
    from collections import Counter
    pc = Counter(py_ids)
    cc = Counter(cpp_ids)
    common = sum((pc & cc).values())
    tp = common
    p = tp / len(cpp_ids) if cpp_ids else 0.0
    r = tp / len(py_ids)  if py_ids  else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def run_cpp_with_memory(save_dir, texts):
    """Run C++ batch demo and measure peak RSS via /usr/bin/time -v.
    Returns (list_of_id_lists, elapsed_sec, peak_rss_kb)."""
    parts = [str(len(texts)).encode("utf-8")]
    for t in texts:
        b = t.encode("utf-8")
        parts.append(str(len(b)).encode("utf-8") + b" " + b)
    stdin_bytes = b"\n".join(parts)

    time_bin = "/usr/bin/time"
    time_args = [time_bin, "-v", TOKENIZE_DEMO, "--stdin-batch", "-t", save_dir]

    t0 = time.perf_counter()
    proc = subprocess.run(time_args, input=stdin_bytes, capture_output=True,
                          timeout=1200)
    elapsed = time.perf_counter() - t0

    # Parse /usr/bin/time -v output on stderr
    peak_kb = 0
    for line in proc.stderr.decode("utf-8", errors="replace").splitlines():
        if "Maximum resident set size" in line:
            peak_kb = int(line.split(":")[1].strip())
            break

    # Parse token IDs from stdout
    results = []
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
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

    while len(results) < len(texts):
        results.append(None)

    return results, elapsed, peak_kb


# ── Report ────────────────────────────────────────────────────────────────

def write_md_and_json(acc_list, lat_list, mismatch_details, langs, en_n, zh_n):
    """Write DATASET_REPORT.md, JSON reports, and mismatch details."""
    # ── Markdown ──
    lines = [
        "# Tokenizer Dataset Accuracy, Latency & Memory Report",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Models tested**: {len(acc_list)}",
        "",
        "## Datasets",
        f"| Language | Dataset | Description | Texts |",
        f"|:---------|:--------|:------------|------:|",
        f"| EN | `{DATASET_META['en']['name']}` | {DATASET_META['en']['description']} | {en_n} |",
        f"| ZH | `{DATASET_META['zh']['name']}` | {DATASET_META['zh']['description']} | {zh_n} |",
        "",
        f"**Source mirror**: {HF_MIRROR}",
        "**Library**: 2.53 MB `libtokenizer_merged.a`",
        "",
        "## 1. Accuracy per Model × Language",
        "",
    ]
    hdr = "| Model |" + "".join(f" {l[:14]} |" for l in langs) + " Token F1 | Overall |"
    sep = "|:------|" + ":----:|" * len(langs) + ":-------:|:-----:|"
    lines.append(hdr); lines.append(sep)
    for a in acc_list:
        if a.get("status") == "SKIP": continue
        row = f"| {a['model']:<12} |"
        for l in langs:
            e = a.get("per_language", {}).get(l, {"ok": 0, "total": 0})
            p = e["ok"] / e["total"] * 100 if e["total"] > 0 else 0
            row += f" {p:5.1f}% |"
        f1 = a.get("token_f1", 0)
        row += f" {f1:.4f} | {a['accuracy_pct']:5.1f}% |"
        lines.append(row)
    n100 = sum(1 for a in acc_list if a.get("accuracy_pct", 0) >= 99.99)
    avg = sum(a.get("accuracy_pct", 0) for a in acc_list
              if a.get("status") != "SKIP") / max(len(acc_list), 1)
    avg_f1 = sum(a.get("token_f1", 0) for a in acc_list
                 if a.get("status") != "SKIP") / max(len(acc_list), 1)
    lines.append(f"\n**100% models**: {n100}/{len(acc_list)} | **Avg accuracy**: {avg:.1f}% | **Avg Token F1**: {avg_f1:.4f}\n")

    # Latency
    lines.append("## 2. Latency & Memory per Model\n")
    lines.append("| Model | C++ (ms) | Python (ms) | Speedup | Peak RSS (KB) | Peak RSS (MB) |")
    lines.append("|:------|--------:|-----------:|-------:|-------------:|-------------:|")
    for l2 in lat_list:
        if l2.get("status") == "SKIP": continue
        mem_kb = l2.get("cpp_peak_rss_kb", 0)
        mem_mb = mem_kb / 1024
        lines.append(
            f"| {l2['model']:<12} | {l2['cpp_avg_latency_ms']:>7.3f} | "
            f"{l2['python_avg_latency_ms']:>10.3f} | {l2['cpp_vs_python_speedup']:>5.1f}x | "
            f"{mem_kb:>12,} | {mem_mb:>11.1f} |")

    # Summary
    ok = sum(a.get("correct", 0) for a in acc_list if a.get("status") != "SKIP")
    mm = sum(a.get("mismatches", 0) for a in acc_list if a.get("status") != "SKIP")
    g = ok + mm
    lines.append("## 3. Summary\n")
    lines.append(f"- **Total comparisons**: {g:,} (model × text)")
    lines.append(f"- **Exact matches**: {ok:,} ({ok/g*100:.1f}%)")
    lines.append(f"- **Mismatches**: {mm:,} ({mm/g*100:.1f}%)")
    avg_mem = sum(l2.get("cpp_peak_rss_kb", 0) for l2 in lat_list if l2.get("status") != "SKIP" and l2.get("cpp_peak_rss_kb", 0) > 0) / max(len(acc_list), 1)
    lines.append(f"- **Avg C++ peak RSS**: {avg_mem:,.0f} KB ({avg_mem/1024:.1f} MB)")
    if any(len(md) > 0 for _, md in mismatch_details):
        lines.append(f"- **Mismatch details**: saved separately to `token_mismatch_details.json`")

    md_path = os.path.join(RESULT_DIR, "DATASET_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✓ Report: {md_path}")

    # ── JSON mismatch details (separate file, not in MD report) ──
    mm_path = os.path.join(RESULT_DIR, "token_mismatch_details.json")
    mm_data = {name: samples for name, samples in mismatch_details if samples}
    with open(mm_path, "w", encoding="utf-8") as f:
        json.dump(mm_data, f, indent=2, ensure_ascii=False)
    if mm_data:
        print(f"✓ Mismatch details: {mm_path}")

# ── Checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CKPT_FILE):
        try:
            data = json.load(open(CKPT_FILE, encoding="utf-8"))
            return (set(data.get("done", [])),
                    data.get("acc", []), data.get("lat", []),
                    data.get("mm", []))
        except Exception:
            pass
    return set(), [], [], []


def save_checkpoint(done_names, acc_list, lat_list, mm_list):
    tmp = CKPT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done_names), "acc": acc_list,
                   "lat": lat_list, "mm": mm_list}, f, ensure_ascii=False)
    os.replace(tmp, CKPT_FILE)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    done_set, acc_rpt, lat_rpt, mm_rpt = load_checkpoint()

    print("=" * 70 + "\n  ZOO TEST 4 — Dataset Accuracy, Latency & Memory Benchmark\n" + "=" * 70)
    if done_set:
        print(f"  [CKPT] {len(done_set)} model(s) already done, resuming ...")

    N_EN = 20000
    N_ZH = 20000

    print(f"\n[DS] Downloading from {HF_MIRROR} ...")
    try:
        en_texts = download_texts_en(N_EN)
        print(f"  ✓ {DATASET_META['en']['name']}: {len(en_texts)} texts")
    except Exception as e:
        print(f"  ✗ EN download failed: {e}"); return 1

    try:
        zh_texts = download_texts_zh(N_ZH)
        print(f"  ✓ {DATASET_META['zh']['name']}: {len(zh_texts)} texts")
    except Exception as e:
        print(f"  ✗ ZH download failed: {e}"); return 1

    en_chars = sum(len(t) for t in en_texts)
    zh_chars = sum(len(t) for t in zh_texts)
    print(f"  EN: {en_chars:,} chars  |  ZH: {zh_chars:,} chars  |  Total: {en_chars+zh_chars:,} chars\n")

    langs = ["en (TinyStories)", "zh (Alpaca-ZH)"]
    texts_by_lang = {langs[0]: en_texts, langs[1]: zh_texts}
    all_texts = en_texts + zh_texts
    same_as = {}

    # mm_rpt: list of (model_name, [mismatch_dicts])
    # Convert checkpoint-loaded mm_rpt to dict for fast lookup
    mm_by_model = {m["model"]: m.get("samples", [])
                   for m in mm_rpt if isinstance(m, dict)}

    try:
        for idx_all, (name, (mid, _, extra)) in enumerate(MODELS.items()):
            if name in done_set:
                print(f"\n[{idx_all+1:>2}/{len(MODELS)}] {name} — already done, skipped")
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
                acc_rpt.append({"model": name, "status": "SKIP"})
                lat_rpt.append({"model": name, "status": "SKIP"})
                mm_by_model[name] = []
                done_set.add(name); save_checkpoint(done_set, acc_rpt, lat_rpt, mm_rpt)
                continue

            sd = f"/tmp/rknn3_zoo_test_{name}"
            hf.save_pretrained(sd)
            if kw.pop("strip_spm", False):
                spm = os.path.join(sd, "tokenizer.model")
                if os.path.exists(spm): os.remove(spm)
            use_saved = kw.pop("use_saved_tok", False)
            py = load_saved_tokenizer(sd) if use_saved else hf

            # ── C++ batch + memory measurement ──
            cpp_results, batch_cpp_time, peak_kb = run_cpp_with_memory(sd, all_texts)
            if batch_cpp_time <= 0: batch_cpp_time = 0.001

            # ── Python batch ──
            t_start = time.perf_counter()
            py_ids_all = [get_py_ids(py, t) for t in all_texts]
            py_batch_time = time.perf_counter() - t_start

            ok, fail = 0, 0
            sum_precision, sum_recall, sum_f1 = 0.0, 0.0, 0.0
            pl = defaultdict(lambda: {"ok": 0, "total": 0})
            total_all = len(all_texts)
            mismatches = []

            done = 0
            for label, lt in texts_by_lang.items():
                for i in range(len(lt)):
                    pids = py_ids_all[done]
                    cids = cpp_results[done] if done < len(cpp_results) else None
                    pl[label]["total"] += 1

                    # Accumulate token-level F1 for every text
                    pr, rc, f1 = compute_f1_score(pids, cids)
                    sum_precision += pr
                    sum_recall    += rc
                    sum_f1        += f1

                    if cids is None:
                        fail += 1
                    elif cids == pids or (len(cids) == 0 and len(pids) == 0):
                        ok += 1; pl[label]["ok"] += 1
                    else:
                        fail += 1
                        if len(mismatches) < 50:  # cap at 50 samples
                            mismatches.append({
                                "lang": label, "text": all_texts[done],
                                "py_ids": pids, "cpp_ids": cids,
                                "precision": round(pr, 4),
                                "recall":    round(rc, 4),
                                "f1":        round(f1, 4)})

                    done += 1
                    if done % 500 == 0 or done == total_all:
                        pct = done / total_all * 100
                        bar = "#" * int(pct/2) + "-" * (50 - int(pct/2))
                        print(f"\r   [{bar}] {done}/{total_all} ({pct:.0f}%)", end="")

            print()
            tot = ok + fail
            acc = ok / tot * 100 if tot else 0
            avg_f1 = sum_f1 / tot if tot else 0
            cms = batch_cpp_time / tot * 1000 if tot else 0
            pms = py_batch_time / tot * 1000 if tot else 0
            sp = pms / cms if cms else 0

            print(f"   Acc: {acc:.2f}%  F1: {avg_f1:.4f}  ({ok}/{tot})  C++ {cms:.2f}ms  Py {pms:.2f}ms  {sp:.1f}x  RSS: {peak_kb/1024:.1f}MB")
            for l in langs:
                e = pl[l]; p = e["ok"] / e["total"] * 100 if e["total"] else 0
                m = "✓" if p >= 99.99 else ("⚠" if p >= 90 else "✗")
                print(f"     {m} {l:<24} {p:5.1f}% ({e['ok']}/{e['total']})")

            acc_rpt.append({
                "model": name, "model_id": mid, "status": "PASS",
                "total_texts": tot, "correct": ok, "mismatches": fail,
                "accuracy_pct": round(acc, 2),
                "token_f1": round(avg_f1, 4),
                "per_language": {k: {"ok": v["ok"], "total": v["total"]}
                                 for k, v in pl.items()},
            })
            lat_rpt.append({
                "model": name, "cpp_avg_latency_ms": round(cms, 4),
                "python_avg_latency_ms": round(pms, 4),
                "cpp_vs_python_speedup": round(sp, 2),
                "cpp_peak_rss_kb": peak_kb,
            })

            mm_by_model[name] = mismatches
            done_set.add(name)

            # Write intermediate reports after each model
            mm_rpt_serialisable = [{"model": k, "samples": v}
                                    for k, v in mm_by_model.items()]
            save_checkpoint(done_set, acc_rpt, lat_rpt, mm_rpt_serialisable)
            write_md_and_json(acc_rpt, lat_rpt,
                [(k, v) for k, v in mm_by_model.items()],
                langs, N_EN, N_ZH)

    except KeyboardInterrupt:
        print(f"\n\n  ⚠ Interrupted — {len(done_set)} models saved")
        mm_rpt_serialisable = [{"model": k, "samples": v}
                                for k, v in mm_by_model.items()]
        save_checkpoint(done_set, acc_rpt, lat_rpt, mm_rpt_serialisable)
        return 0

    for alias, target in same_as.items():
        for lst in [acc_rpt, lat_rpt]:
            for r in lst:
                if r.get("model") == target:
                    r2 = dict(r); r2["model"] = alias; lst.append(r2); break
        if target in mm_by_model:
            mm_by_model[alias] = mm_by_model[target]
        done_set.add(alias)

    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)

    with open(os.path.join(RESULT_DIR, "token_acc_dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(acc_rpt, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULT_DIR, "token_latency_report.json"), "w", encoding="utf-8") as f:
        json.dump(lat_rpt, f, indent=2, ensure_ascii=False)
    write_md_and_json(acc_rpt, lat_rpt,
        [(k, v) for k, v in mm_by_model.items()],
        langs, N_EN, N_ZH)

    ok_t = sum(a.get("correct", 0) for a in acc_rpt if a.get("status") != "SKIP")
    mm_t = sum(a.get("mismatches", 0) for a in acc_rpt if a.get("status") != "SKIP")
    print(f"\n{'='*70}\n  TOTAL: {ok_t:,}/{ok_t+mm_t:,} correct ({ok_t/(ok_t+mm_t)*100:.1f}%)\n{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

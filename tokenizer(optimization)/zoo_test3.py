#!/usr/bin/env python3
"""Deep alignment test — exhaustive edge-case coverage across all models."""

import os
import subprocess
import sys

TOKENIZE_DEMO = os.path.join(os.path.dirname(__file__),
    "build", "build_tokenizer_linux_x86_Release", "demo", "tokenize_demo")

MODELS = {
    "Qwen3":           ("Qwen/Qwen3-1.7B",              None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen2_5":         ("Qwen/Qwen2.5-3B-Instruct",     None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen2_5_VL":      ("Qwen/Qwen2.5-VL-3B-Instruct",  None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen2_5_Omni":    ("Qwen/Qwen2.5-Omni-3B",         None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen3_VL":        ("Qwen/Qwen3-VL-4B-Instruct",     None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen3_VL_LoRA":   ("Qwen/Qwen3-VL-4B-Instruct",     None,
                       {"_same_as": "Qwen3_VL"}),
    "Qwen3_ASR":       ("Qwen/Qwen3-ASR-0.6B",          None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen3_Embedding": ("Qwen/Qwen3-Embedding-0.6B",    None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen3_Reranker":  ("Qwen/Qwen3-Reranker-0.6B",     None,
                       {"use_saved_tok_for_comparison": True}),
    "Qwen3_TTS":       ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", None,
                       {"use_saved_tok_for_comparison": True}),
    "HY_MT_1_5":       ("Tencent-Hunyuan/HY-MT1.5-1.8B", None,
                        {"use_modelscope": True}),
    "Janus_Pro":       ("deepseek-ai/Janus-Pro-1B",      None,
                        {"use_saved_tok_for_comparison": True}),
    "SmolVLM":         ("HuggingFaceTB/SmolVLM-500M-Instruct", None, None),
    "SmolVLM2":        ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", None, None),
    "glm_edge":        ("THUDM/glm-edge-1.5b-chat",     None, None),
    "GME-Qwen2-VL":    ("Alibaba-NLP/GME-Qwen2-VL-2B-Instruct", None,
                       {"use_saved_tok_for_comparison": True}),
    "gemma4":          ("google/gemma-4-E2B-it",         None,
                        {"trust_remote_code": True, "use_fast": False,
                         "use_saved_tok_for_comparison": True}),
    "InternVLM":       ("OpenGVLab/InternVL3-1B",        None,
                       {"trust_remote_code": True,
                        "use_saved_tok_for_comparison": True}),
    "MiniCPM_V_4":     ("openbmb/MiniCPM-V-4",           None,
                       {"trust_remote_code": True,
                        "use_saved_tok_for_comparison": True,
                        "strip_spm": True}),
    "FastVLM":         ("llava-hf/llava-1.5-7b-hf",      None, None),
}

# ─── Test case definitions ─────────────────────────────────────────────────

TEST_CASES = {
    # ==================== 1. 中英文混合 ====================
    "1. 中英文混合": [
        "Hello世界",
        "你好World",
        "他说hello然后走了",
        "Python是人工智能领域的首选语言。",
        "The word 人工智能 means artificial intelligence.",
        "我在Google工作了很多年。",
        "今天天气非常好sunny而且warm。",
        "代码review完了，请merge到main分支。",
        "请使用pip install transformers安装依赖。",
        "API接口返回了404 Not Found错误。",
        "a你好b世界c",
        "2024年AI发展迅速，GPT-4o和Claude都很强。",
        "Москва是俄罗斯的首都。",
        "東京で寿司を食べる",
    ],

    # ==================== 2. 标点符号 ====================
    "2. 标点符号": [
        "Hello, world! How are you?",
        "他说：'你好'。",
        "价格：¥99.99（含税）",
        "《红楼梦》是中国古典文学名著。",
        "a—b…c、d。e·f",
        "Q1: 什么是AI？A1: 人工智能。",
        "!!!???...",
        "### 标题 ###",
        "价格是$100～$200/个",
        "用户@admin: 请确认#123号工单。",
        "()[]{}<>《》「」【】",
        '"单引号\'和双引号"的混合',
        "3 * 4 = 12; 10 / 2 = 5; x += 1;",
        "~`!@#$%^&*()-_=+[]{}|;:'\",.<>?/",
    ],

    # ==================== 3. 全角/半角符号 ====================
    "3. 全角/半角符号": [
        "ＡＢＣＤＥＦＧ",
        "ａｂｃｄｅｆｇ",
        "１２３４５６",
        "，。！？；：＂＇",
        "Hello，世界。",
        "全角１２３和半角123混合",
        "【重要】全角括号测试",
        "０１２３４５６７８９",
        "℡℃㎡№",
        "① ② ③",
        "半角abc全角ａｂｃ混排。",
        "12.5kg＝12,500g",
        "ｱｲｳｴｵ（半角片假名）",
    ],

    # ==================== 4. 长文本连续性 ====================
    "4. 长文本连续性": [
        "从前有一只狐狸，它跳过了一只懒狗。" * 20,
        "a" * 1000,
        "Hello world. " * 200,
        "人工智能是计算机科学的一个分支。" * 30,
        "".join(chr(0x4E00 + i % 20000) for i in range(0, 500, 7)),
        "The " * 500 + "end",
    ],

    # ==================== 5a. 空字符串 ====================
    "5a. 空字符串": [""],

    # ==================== 5b. 极端重复 ====================
    "5b. 极端重复": [
        "测试" * 500,
        "hello" * 500,
        "。" * 200,
        "a " * 500,
        "12" * 400,
    ],

    # ==================== 5c. 生僻字/繁体字 ====================
    "5c. 生僻字/繁体字": [
        "龘靐齉爨癵籱饢驫",
        "𠀀𠀁𠀂𠀃𠀄",
        "繁體字測試：壹貳參肆伍陸柒捌玖拾",
        "生僻成語：夔魖鼉鼇",
        "化學元素：氫氦鋰鈹硼碳氮氧氟氖",
        "康熙字典字：𪚥",
        "䶵𤴐𪛖𪟠",
    ],

    # ==================== 5d. 换行/制表符/空格 ====================
    "5d. 换行/制表符/空格": [
        "a\nb\nc",
        "a\tb\tc",
        "a\r\nb\r\nc",
        "a\f\vb",
        "    a    b    ",
        "\n\n\n",
        "\t\t\t",
        "混合 空格\t制表\n换行\r回车\f换页",
        "a b c",
        "a    b",
    ],

    # ==================== 5e. 超长单句 ====================
    "5e. 超长单句": [
        "人工智能作为计算机科学的一个重要分支，近年来在深度学习技术的推动下取得了突破性进展，尤其是在自然语言处理、计算机视觉和强化学习等领域展现出了强大的能力，正在深刻地改变着人们的生产和生活方式。",
        "The quick brown fox jumps over the lazy dog, which happens to be one of the most famous pangrams in the English language, as it contains every single letter of the alphabet at least once, and has been used for typing practice and font testing for well over a century.",
    ],

    # ==================== 5f. 不可见特殊 Unicode ====================
    "5f. 不可见特殊 Unicode": [
        "hello​world",
        "﻿​‌‍⁠",
        "à́̂̃b",
        "hello⁣world",
        "‎‏؜",
        "café",
        "‮‭",
    ],

    # ==================== 6. 协议/代码/JSON ====================
    "6. 协议/代码/JSON": [
        '{"name": "Alice", "age": 30, "city": "北京"}',
        "<html><body><h1>标题</h1><p>段落</p></body></html>",
        "def hello(name):\n    return f'Hello, {name}!'",
        "SELECT * FROM users WHERE city = '上海' AND age > 18;",
        "https://example.com/搜索?q=人工智能&lang=zh",
        "```python\nprint('Hello, 世界!')\n```",
        "| 列A | 列B |\n|-----|-----|\n| 数据1 | 数据2 |",
        "2024-08-04T15:30:00+08:00",
        "printf('%s: %d\n', name, count);",
        "{\\} {\\} 123",
    ],

    # ==================== 7. Emoji/特殊符号 ====================
    "7. Emoji/特殊符号": [
        "Hello \U0001F600 World \U0001F30D",
        "\U0001F44D\U0001F44E\U0001F91E\U0001F4AF\U0001F525\U0001F389",
        "数学符号：∑∫√∞≈≠≤≥",
        "箭头：→←↑↓↔⇒⇐⇑⇓⟹",
        "\U0001F600\U0001F602\U0001F923\U0001F60A\U0001F60D\U0001F618",
        "货币符号：$ € £ ¥ ₩ ₹ ₿",
    ],

    # ==================== 8. 数字格式 ====================
    "8. 数字格式": [
        "3.14159265358979323846",
        "-273.15°C is absolute zero",
        "1,000,000 vs 1000000",
        "0xDEADBEEF and 0b10101010",
        "第1章 第2节 第3小节",
        "Ⅱ Ⅲ Ⅳ Ⅴ Ⅵ Ⅶ",
        "99.99% of 1/3 ≈ 0.333...",
        "e = 2.71828..., π = 3.14159...",
    ],
}


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


def parse_cpp_ids(demo_output):
    ids = []
    for line in demo_output:
        line = line.strip()
        if ' -> \'' in line and not line.startswith('Decode'):
            try:
                ids.append(int(line.split(' -> ')[0]))
            except ValueError:
                pass
    return ids


def get_py_ids(tok, text):
    r = tok.encode(text, add_special_tokens=False)
    return list(r) if isinstance(r, list) else r.ids


def test_one(py_tok, save_dir, text):
    """Returns (ok, py_ids, cpp_ids, fail_reason) where ok is True/False."""
    py_ids = get_py_ids(py_tok, text)
    try:
        proc = subprocess.run(
            [TOKENIZE_DEMO, "-t", save_dir, "-p", text, "--show-count"],
            capture_output=True, text=True, timeout=30)
        demo_output = proc.stdout.splitlines()
    except Exception as e:
        return (False, py_ids, None, f"demo crashed: {e}")

    load_failed = any("load failed" in l or "Error: could not create"
                      in l for l in demo_output)
    if load_failed:
        return (False, py_ids, None, "load failed")

    cpp_ids = parse_cpp_ids(demo_output)
    if len(cpp_ids) == 0 and len(py_ids) == 0:
        return (True, py_ids, cpp_ids, "")
    ok = cpp_ids == py_ids
    reason = "" if ok else "mismatch"
    return (ok, py_ids, cpp_ids, reason)


def main():
    print("=" * 70)
    print("  ZOO TEST 3 — Deep Tokenizer Alignment Test")
    print("=" * 70)
    print(f"  Models:  {len(MODELS)}")
    print(f"  Categories: {len(TEST_CASES)}")
    total_cases = sum(len(v) for v in TEST_CASES.values())
    print(f"  Test cases: {total_cases}")
    print(f"  C++ bin:    {TOKENIZE_DEMO}")
    print("=" * 70)

    # ── Phase 1: load all tokenizers ──
    print("\n[PREP] Loading tokenizers ...")
    loaded = {}
    same_as = {}

    for name, (model_id, _, extra) in MODELS.items():
        kwargs = dict(extra) if extra else {}
        share = kwargs.pop("_same_as", None)
        if share:
            same_as[name] = share
            continue

        try:
            hf = load_tokenizer(model_id, extra)
            save_dir = f"/tmp/rknn3_zoo_test_{name}"
            hf.save_pretrained(save_dir)
            kwargs = dict(extra) if extra else {}
            # strip_spm: delete tokenizer.model so C++ falls back to
            # tokenizer.json (BPE). Used when trust_remote_code=True
            # applies custom Python preprocessing that raw SPM can't
            # replicate.
            if kwargs.pop("strip_spm", False):
                spm_file = os.path.join(save_dir, "tokenizer.model")
                if os.path.exists(spm_file):
                    os.remove(spm_file)
            use_saved = kwargs.pop("use_saved_tok_for_comparison", False)
            py_tok = load_saved_tokenizer(save_dir) if use_saved else hf
            loaded[name] = {"py_tok": py_tok, "save_dir": save_dir}
            print(f"  ✓  {name}")
        except Exception as e:
            loaded[name] = {"error": str(e)[:80]}
            print(f"  ⏭  {name}  —  {e}")

    # Resolve _same_as aliases
    for alias, target in same_as.items():
        if target in loaded and "error" not in loaded[target]:
            loaded[alias] = loaded[target]
            print(f"  ✓  {alias}  (→ {target})")
        else:
            loaded[alias] = {"error": f"_same_as target '{target}' not loaded"}
            print(f"  ⏭  {alias}  —  _same_as target '{target}' not loaded")

    # ── Phase 2: run every category ──
    all_results = {}   # {cat_name: {model_name: (n_pass, n_fail)}}
    all_failures = {}  # {cat_name: {model_name: [(text, py_ids, cpp_ids, reason), ...]}}
    grand_pass = 0
    grand_total = 0

    for cat_name, cat_texts in TEST_CASES.items():
        print(f"\n{'─'*70}")
        print(f"  📁 {cat_name}  ({len(cat_texts)} cases)")
        print(f"{'─'*70}")
        print(f"  {'Model':<20}  {'PASS':>6}  {'FAIL':>6}  Status")
        print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*6}")

        cat_results = {}
        cat_failures = {}
        n_pass_models = 0
        n_fail_models = 0

        for name in MODELS:
            entry = loaded.get(name, {})
            if "error" in entry:
                print(f"  {name:<20}  {'─':>6}  {'─':>6}  SKIP")
                cat_results[name] = (0, 0)
                cat_failures[name] = []
                continue

            p = 0
            f = 0
            failures = []
            for text in cat_texts:
                ok, py_ids, cpp_ids, reason = test_one(
                    entry["py_tok"], entry["save_dir"], text)
                if ok:
                    p += 1
                else:
                    f += 1
                    failures.append((text, py_ids, cpp_ids, reason))

            cat_results[name] = (p, f)
            cat_failures[name] = failures
            total = p + f
            grand_pass += p
            grand_total += total

            if f == 0:
                status = "✓"
                n_pass_models += 1
            else:
                status = "✗"
                n_fail_models += 1
            print(f"  {name:<20}  {p:>6}  {f:>6}  {status}")

        all_results[cat_name] = cat_results
        all_failures[cat_name] = cat_failures
        print(f"  {'─'*20}  {'─'*6}  {'─'*6}")
        print(f"  {'':20}  ✓ {n_pass_models} models pass   ✗ {n_fail_models} models fail")

        # Print failure details for this category
        for name in MODELS:
            failures = cat_failures.get(name, [])
            if not failures:
                continue
            print(f"\n  {'─'*66}")
            print(f"  ▌ FAIL details: {name}")
            print(f"  {'─'*66}")
            for text, py_ids, cpp_ids, reason in failures:
                text_label = repr(text)
                if len(text_label) > 65:
                    text_label = text_label[:62] + "..."
                print(f"  │ text : {text_label}")
                print(f"  │ HF   : {py_ids}")
                if cpp_ids is not None:
                    print(f"  │ C++  : {cpp_ids}")
                else:
                    print(f"  │ C++  : {reason}")
                print(f"  │")

    # ── Grand summary matrix ──
    print(f"\n{'═'*70}")
    print(f"  GRAND SUMMARY")
    print(f"{'═'*70}")

    n_cats = len(TEST_CASES)
    cat_keys = list(TEST_CASES.keys())

    # Header
    print(f"  {'':20}", end="")
    for i in range(n_cats):
        print(f" {i+1:>2} ", end="")
    print("")

    print(f"  {'─'*20}", end="")
    for i in range(n_cats):
        print(f"───", end="")
    print("")

    # Per-model grid
    model_verdicts = {}
    for name in MODELS:
        entry = loaded.get(name, {})
        if "error" in entry:
            print(f"  {name:<20}  " + "⏭ " * n_cats + " SKIP")
            model_verdicts[name] = "SKIP"
            continue

        print(f"  {name:<20}", end="")
        ok_count = 0
        for cat_name in cat_keys:
            p, f = all_results.get(cat_name, {}).get(name, (0, 0))
            if f == 0:
                print("  ✓", end="")
                ok_count += 1
            else:
                print("  ✗", end="")
        print(f"  {ok_count}/{n_cats}")
        model_verdicts[name] = "PASS" if ok_count == n_cats else "FAIL"

    # Footer
    n_pass = sum(1 for v in model_verdicts.values() if v == "PASS")
    n_fail = sum(1 for v in model_verdicts.values() if v == "FAIL")
    n_skip = sum(1 for v in model_verdicts.values() if v == "SKIP")
    print(f"  {'─'*20}" + "───" * n_cats)
    print(f"  {'':20} PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")

    # Totals
    print(f"\n  {'═'*50}")
    print(f"  Test cases total:    {grand_total:>8}")
    print(f"  C++ == Python:       {grand_pass:>8}")
    print(f"  Mismatches:          {grand_total - grand_pass:>8}")
    if grand_total > 0:
        print(f"  Accuracy:            {grand_pass / grand_total * 100:>7.1f}%")
    print(f"  {'═'*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

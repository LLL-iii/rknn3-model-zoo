#!/usr/bin/env python3
"""gen_test_cases.py — 生成 chat_template 板端测试数据（多场景 × 多变体 ctx + golden）

用法:
  python gen_test_cases.py [--models-dir <tokenizer/models>] [--out <dir>] [--scenarios ...]
  python gen_test_cases.py --list-scenarios          # 列出场景及变体数

对每个有 chat_template 的模型：
  1. 读取模板（chat_template.jinja 或 tokenizer_config.json 的 chat_template 字段）
  2. 按场景注册表生成各场景的多变体 ctx（按模板特性自动过滤适用场景）
  3. 用 Python jinja2（对齐 transformers 环境）渲染 golden
  4. 写 <out>/manifest.json（供板端测试脚本使用）

场景变体命名：ctx/<model>_<scenario>_v<n>.json
每个场景的适用性与变体结构见场景注册表注释。
"""
import argparse
import json
import os
import re

from jinja2 import Environment, nodes
from jinja2.ext import Extension


def _raise_exception(message):
    raise RuntimeError(message)


class GenerationExtension(Extension):
    """HF `{% generation %}` tag — ignored, body rendered as-is."""
    tags = {"generation"}

    def parse(self, parser):
        parser.stream.skip()
        parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.Output([nodes.Const("")])


# ── 基础常量 ──────────────────────────────────────────────────────────────
USER_TXT = "Hello, how are you?"
ASSISTANT_TXT = "I'm doing great! AI is the simulation of human intelligence in machines. Let me elaborate."
SYSTEM_TXT = "You are a helpful assistant."
LONG_PARA = "The quick brown fox jumps over the lazy dog. "

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name"},
                                      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Get current time.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _mm_content(text, is_multimodal):
    """多模态模板（SmolVLM 等）content 为 list of dict；文本模板为字符串。
    已传 list（multimodal 变体直接给 [{"type":"text"/"image"},...]）时不重复包装。"""
    if is_multimodal and not isinstance(text, list):
        return [{"type": "text", "text": text}]
    return text


def is_multimodal_template(src):
    return bool(re.search(r"line\['type'\]|line\[\"type\"\]|content\]\[0\]|content\]\['type'\]", src))


def _msg(role, content, is_mm, **kw):
    """构造消息；content 按多模态(list)/文本(string)格式；kw 追加额外字段（tool_calls 等）。"""
    m = {"role": role, "content": _mm_content(content, is_mm)}
    m.update(kw)
    return m


def _tc(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# ── 场景注册表 ────────────────────────────────────────────────────────────
# name -> {"need": None|函数(src)->bool, "desc": str, "variants": (bos,eos,is_mm)->[ctx]}
SCENARIOS = {}


def _reg(name, need, desc, fn):
    SCENARIOS[name] = {"need": need, "desc": desc, "variants": fn}


def _need_tools(src): return "tools" in src
def _need_system(src): return "system" in src
def _need_mm(src): return is_multimodal_template(src)


# ---- 已有场景（细化消息结构定义）----
_reg("single", None, "仅单条 user（不含 system）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", USER_TXT, mm)],
     "add_generation_prompt": True},
])

_reg("multi", None, "system + 2 轮问答", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "What is artificial intelligence?", mm),
                  _msg("assistant", ASSISTANT_TXT, mm),
                  _msg("user", "Tell me more about it.", mm)],
     "add_generation_prompt": True},
])

_reg("nogen", None, "仅 user，不加生成标记", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", USER_TXT, mm)],
     "add_generation_prompt": False},
])

_reg("long", None, "system + 6 轮（loop 迭代边界）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Q1: What is 1+1?", mm), _msg("assistant", "A1: It is 2.", mm),
                  _msg("user", "Q2: What is 2+2?", mm), _msg("assistant", "A2: It is 4.", mm),
                  _msg("user", "Q3: What is 3+3?", mm)],
     "add_generation_prompt": True},
])

_reg("empty", None, "空 messages（messages[0] 越界）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos, "messages": [], "add_generation_prompt": False},
])

_reg("longtext", None, "200k 长文本（含上下文）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Please summarize the following long document.", mm),
                  _msg("user", (LONG_PARA * (200000 // len(LONG_PARA) + 1))[:200000], mm)],
     "add_generation_prompt": True},
])

_reg("tools", _need_tools, "单次工具调用闭环（arguments=dict）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "What's the weather in Beijing?", mm),
                  _msg("assistant", "", mm, tool_calls=[_tc("call_0", "get_weather", {"city": "Beijing"})]),
                  _msg("tool", '{"temperature": 25, "condition": "sunny"}', mm, tool_call_id="call_0"),
                  _msg("assistant", "It's 25 degrees and sunny in Beijing.", mm)],
     "add_generation_prompt": True},
])

_reg("thinking", None, "思考链（enable_thinking true/false）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos, "enable_thinking": True, "preserve_thinking": False,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Solve: 17 * 23 = ?", mm),
                  _msg("assistant", "The answer is 391.", mm,
                       reasoning_content="Let me compute: 17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391."),
                  _msg("user", "Double-check please.", mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos, "enable_thinking": False,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Hi.", mm),
                  _msg("assistant", "Hello!", mm, reasoning_content="thinking...")],
     "add_generation_prompt": True},
])

# ---- 新场景 ----
# 1. 系统消息变更：system 可能动态更新（任务指令替换），也可能缺失/为空
_reg("system_dynamic", _need_system, "system 动态更新 / 缺失 / 为空", lambda bos, eos, mm: [
    # v1: 自定义 system 内容（动态更新场景：每次请求传不同 system）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", "You are a code assistant. Always answer with code only.", mm),
                  _msg("user", "Write a hello world.", mm),
                  _msg("assistant", "```python\nprint('hello')\n```", mm),
                  _msg("user", "Now in C++.", mm)],
     "add_generation_prompt": True},
    # v2: 无 system（messages[0]=user）→ 模板默认 system 分支
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "Hello, what can you do?", mm),
                  _msg("assistant", "I can help with many tasks.", mm)],
     "add_generation_prompt": True},
    # v3: system 为空字符串
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", "", mm),
                  _msg("user", "Hi there.", mm)],
     "add_generation_prompt": True},
])

# 2. 多角色混合：连续多次工具调用 + 最终回答的完整闭环
_reg("multi_tool", _need_tools, "连续多次工具调用 + 最终回答", lambda bos, eos, mm: [
    # v1: assistant 一次发起 2 个 tool_calls → 2 个 tool 响应 → 最终回答
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Weather in Beijing and current time?", mm),
                  _msg("assistant", "", mm,
                       tool_calls=[_tc("c1", "get_weather", {"city": "Beijing"}),
                                   _tc("c2", "get_time", {})]),
                  _msg("tool", '{"temperature": 25, "condition": "sunny"}', mm, tool_call_id="c1"),
                  _msg("tool", '{"time": "14:30"}', mm, tool_call_id="c2"),
                  _msg("assistant", "It's 25 degrees and sunny in Beijing, current time 14:30.", mm)],
     "add_generation_prompt": True},
    # v2: 两轮 tool 重试（调用失败 → tool 响应 → assistant 再次调用 → 最终回答）
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Get weather in Tokyo.", mm),
                  _msg("assistant", "", mm, tool_calls=[_tc("t1", "get_weather", {"city": "Tokyo"})]),
                  _msg("tool", '{"error": "city not found"}', mm, tool_call_id="t1"),
                  _msg("assistant", "", mm, tool_calls=[_tc("t2", "get_weather", {"city": "Tokyo", "unit": "celsius"})]),
                  _msg("tool", '{"temperature": 20, "condition": "cloudy"}', mm, tool_call_id="t2"),
                  _msg("assistant", "It's 20 degrees and cloudy in Tokyo.", mm)],
     "add_generation_prompt": True},
])

# 3. 特殊字符与转义：换行/引号/反斜杠/Unicode/HTML
_reg("special_chars", None, "换行/引号/反斜杠/Unicode/HTML 特殊字符", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", 'Line1\nLine2 "quoted" \\backslash\\ and \'single\'.', mm),
                  _msg("assistant", 'Respond with: "ok\\n".', mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "中文文本：你好，世界！Emoji: 😀🚀 混合 English 123.", mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "HTML: <b>bold</b> & <i>italic</i> \"quotes\" \'apostrophes\'.", mm)],
     "add_generation_prompt": True},
])

# 4. 空 assistant / 结尾缺失：末尾角色不完整时模板行为
_reg("assistant_end_missing", None, "末尾为 user/tool/空 assistant 的边界", lambda bos, eos, mm: [
    # v1: 以 user 结尾 + generation prompt（正常生成）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Question 1?", mm),
                  _msg("assistant", "Answer 1.", mm),
                  _msg("user", "Question 2?", mm)],
     "add_generation_prompt": True},
    # v2: 以 tool 结尾（无最终 assistant；仅 tools 模板有意义，其余跳过）
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Call weather.", mm),
                  _msg("assistant", "", mm, tool_calls=[_tc("c9", "get_weather", {"city": "Paris"})]),
                  _msg("tool", '{"temperature": 18}', mm, tool_call_id="c9")],
     "add_generation_prompt": True},
    # v3: assistant 空内容消息
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Hi.", mm),
                  _msg("assistant", "", mm),
                  _msg("user", "Are you there?", mm)],
     "add_generation_prompt": True},
])

# 5. 多模态输入：图像占位符位置与替换（仅视觉模型）
_reg("multimodal", _need_mm, "图像占位符（视觉模型）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", "You are a vision assistant.", True),
                  _msg("user", [{"type": "text", "text": "Describe this image."}, {"type": "image"}], True)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", [{"type": "text", "text": "Compare:"}, {"type": "image"},
                                {"type": "text", "text": "and"}, {"type": "image"}], True)],
     "add_generation_prompt": True},
])

# 6. 模板变量插值：额外变量（date/user_name 等），验证传入模板未用到的变量不报错
_reg("template_vars", None, "额外变量（date/user_name 等）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos, "date": "2026-08-20", "user_name": "Alice",
     "messages": [_msg("user", "What date is it, Alice?", mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos, "current_date": "Monday", "weather": "sunny",
     "messages": [_msg("user", "Hello.", mm)],
     "add_generation_prompt": True},
])

# 7. 超长上下文：30 轮历史（模板不做截断；截断在 LLM/服务层，此处测多轮长上下文渲染正确与性能）
_reg("long_hist", None, "30 轮历史（超长上下文多轮）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm)] +
                 [m for i in range(30)
                  for m in (_msg("user", f"Q{i}: value {i}?", mm), _msg("assistant", f"A{i}: result {i}.", mm))],
     "add_generation_prompt": True},
])

# 8. 只有 system 消息
_reg("system_only", _need_system, "仅 system 消息", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", "You are a strict validator. Never improvise.", mm)],
     "add_generation_prompt": False},
])

# 9. 复杂角色交替：非标准顺序（user/assistant/tool），检查模板是否强制顺序
_reg("role_alternation", None, "非标准顺序交替（user/tool/assistant）", lambda bos, eos, mm: [
    # v1: user→assistant→tool→user→assistant（tool 后接 user）
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Help me.", mm),
                  _msg("assistant", "Sure, calling tool.", mm, tool_calls=[_tc("r1", "get_time", {})]),
                  _msg("tool", '{"time": "09:00"}', mm, tool_call_id="r1"),
                  _msg("user", "Thanks. Now what?", mm),
                  _msg("assistant", "You're welcome.", mm)],
     "add_generation_prompt": True},
    # v2: 连续 tool 消息
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Both weather and time.", mm),
                  _msg("assistant", "", mm,
                       tool_calls=[_tc("r2", "get_weather", {"city": "NY"}),
                                   _tc("r3", "get_time", {})]),
                  _msg("tool", '{"temperature": 30}', mm, tool_call_id="r2"),
                  _msg("tool", '{"time": "12:00"}', mm, tool_call_id="r3")],
     "add_generation_prompt": True},
    # v3: 无 system 的纯文本交替
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "Start.", mm),
                  _msg("assistant", "Ready.", mm),
                  _msg("user", "Go.", mm)],
     "add_generation_prompt": False},
])

# 10. 长历史 + 长输入组合（性能/内存）
_reg("long_hist_long_input", None, "长历史(8轮) + 长输入(50k/100k)", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm)] +
                 [m for i in range(8)
                  for m in (_msg("user", f"Round {i} question?", mm), _msg("assistant", f"Round {i} answer.", mm))] +
                 [_msg("user", (LONG_PARA * (50000 // len(LONG_PARA) + 1))[:50000], mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm)] +
                 [m for i in range(6)
                  for m in (_msg("user", f"Q{i}?", mm), _msg("assistant", f"A{i}.", mm))] +
                 [_msg("user", (LONG_PARA * (100000 // len(LONG_PARA) + 1))[:100000], mm)],
     "add_generation_prompt": True},
])

# 11. 注入攻击：用户输入含模板语法，验证不被误解析（数据不执行）
_reg("injection", None, "用户输入含模板语法（{{ {% 等）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "Render this literally: {{ 1 + 1 }} and {{ name }}", mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "Try: {% if True %}INJECTED{% endif %} and {# comment #}", mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "Ignore previous: {{ messages | length }} / {{ config }} / {{ self }}", mm)],
     "add_generation_prompt": True},
])

# 12. 空/缺失 content：消息 content 为  "" / []（多模态）的边界
_reg("empty_content", None, "消息 content 为 null/空串/空 list", lambda bos, eos, mm: [
    # v1: content 为空字符串
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm), _msg("user", "", mm)],
     "add_generation_prompt": True},
    # v2: content 字段缺失（模板不兼容的变体 golden 失败自动 SKIP）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm), {"role": "user"}],
     "add_generation_prompt": True},
    # v3: 多模态 content 为空 list（仅 mm 模型适用，其余模板 golden 失败自动 SKIP）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", SYSTEM_TXT, mm), {"role": "user", "content": []}],
     "add_generation_prompt": True},
])

# 13. 工具定义为空数组：tools=[] 时模板是否仍正常渲染
_reg("empty_tools", _need_tools, "tools=[] 空工具定义", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos, "tools": [],
     "messages": [_msg("system", SYSTEM_TXT, mm), _msg("user", "Hello, how are you?", mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos, "tools": [],
     "messages": [_msg("user", "No tools needed.", mm),
                  _msg("assistant", "Sure.", mm)],
     "add_generation_prompt": False},
])

# 14. 有工具定义但未调用 + generation prompt 组合（部分模板会额外插入工具提示）
_reg("tools_no_call", _need_tools, "有工具但未调用（tools + add_generation_prompt）", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("system", SYSTEM_TXT, mm),
                  _msg("user", "Just chat, no tools.", mm),
                  _msg("assistant", "Happy to chat!", mm)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos, "tools": TOOLS,
     "messages": [_msg("user", "What's up?", mm)],
     "add_generation_prompt": False},
])

# 15. 多模态 image 占位符带 URL / base64（仅视觉模型）
_reg("multimodal_image_url", _need_mm, "多模态 image 带 URL/base64", lambda bos, eos, mm: [
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", "You are a vision assistant.", True),
                  _msg("user", [{"type": "text", "text": "What's in this image?"},
                                {"type": "image", "image_url": "https://example.com/img.png"}], True)],
     "add_generation_prompt": True},
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", [{"type": "text", "text": "Describe:"},
                                {"type": "image", "data": "iVBORw0KGgoAAAANSUhEUg=="}], True)],
     "add_generation_prompt": True},
])

# 16. 模板 for/if 复杂逻辑边界（空角色 / 仅末尾 / 推理内容）
_reg("advanced_logic", None, "模板 for/if 分支边界（仅末尾/空角色/推理）", lambda bos, eos, mm: [
    # v1: 仅 user + generation prompt（触发 loop.last 与末尾生成分支）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "Hello.", mm)],
     "add_generation_prompt": True},
    # v2: 仅 assistant（无 generation prompt，非 last 分支）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("assistant", "Just an assistant turn.", mm)],
     "add_generation_prompt": False},
    # v3: assistant 含 reasoning_content（触发 if 推理分支）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("user", "Explain step by step.", mm),
                  _msg("assistant", "Final answer.", mm, reasoning_content="thinking...")],
     "add_generation_prompt": True},
])

# 17. 重复/多条 system
_reg("multi_system", _need_system, "重复/多条 system", lambda bos, eos, mm: [
    # v1: 两条 system（开头 + 中间）
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", "System A.", mm),
                  _msg("user", "Hi.", mm),
                  _msg("system", "System B.", mm),
                  _msg("user", "Continue.", mm)],
     "add_generation_prompt": True},
    # v2: 三条连续 system 开头
    {"bos_token": bos, "eos_token": eos,
     "messages": [_msg("system", "S1.", mm),
                  _msg("system", "S2.", mm),
                  _msg("system", "S3.", mm),
                  _msg("user", "Hello.", mm)],
     "add_generation_prompt": True},
])


def render_golden(src, ctx):
    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        extensions=[GenerationExtension],
    )
    env.globals["raise_exception"] = _raise_exception
    tpl = env.from_string(src)
    return tpl.render(**ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=r"E:\rknn3-model-zoo\rknn3-model-zoo2\tokenizer\models")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS.keys()))
    ap.add_argument("--list-scenarios", action="store_true", help="仅列出场景及变体数")
    args = ap.parse_args()

    if args.list_scenarios:
        for name, sc in SCENARIOS.items():
            print(f"{name:<28} need={sc['need'].__name__ if sc['need'] else '-':<14} "
                  f"variants={len(sc['variants']('', '', False)):<3} {sc['desc']}")
        return

    scenarios = [s for s in args.scenarios.split(",") if s and s in SCENARIOS]
    os.makedirs(os.path.join(args.out, "ctx"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "golden"), exist_ok=True)

    manifest = {"models": [], "scenarios": scenarios}
    for model in sorted(os.listdir(args.models_dir)):
        d = os.path.join(args.models_dir, model)
        cfg_path = os.path.join(d, "tokenizer_config.json")
        jinja_path = os.path.join(d, "chat_template.jinja")
        if not os.path.exists(cfg_path):
            continue
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8"))
        except Exception:
            continue
        src = None
        tpl_file = None
        if os.path.exists(jinja_path):
            src = open(jinja_path, encoding="utf-8").read()
            tpl_file = "chat_template.jinja"
        else:
            ct = cfg.get("chat_template")
            if isinstance(ct, str):
                src = ct
                tpl_file = "tokenizer_config.json"
            elif isinstance(ct, dict) and isinstance(ct.get("default"), str):
                src = ct["default"]
                tpl_file = "tokenizer_config.json"
        if src is None or tpl_file is None:
            continue  # 无模板

        bos = cfg.get("bos_token", "")
        eos = cfg.get("eos_token", "")
        is_mm = is_multimodal_template(src)

        cases = []
        for sc_name, sc in SCENARIOS.items():
            if sc_name not in scenarios:
                continue
            need = sc["need"]
            if need and not need(src):
                continue
            for vi, ctx in enumerate(sc["variants"](bos, eos, is_mm), 1):
                try:
                    golden = render_golden(src, ctx)
                except Exception as e:
                    print(f"  SKIP: {model} {sc_name}_v{vi} golden 渲染失败（模板不适应该变体）: {str(e)[:60]}")
                    continue
                sc_full = f"{sc_name}_v{vi}"
                ctx_path = os.path.join(args.out, "ctx", f"{model}_{sc_full}.json")
                with open(ctx_path, "wb") as f:
                    f.write(json.dumps(ctx, ensure_ascii=False).encode("utf-8"))
                golden_path = os.path.join(args.out, "golden", f"{model}_{sc_full}.golden")
                with open(golden_path, "wb") as f:
                    f.write(golden.encode("utf-8"))
                cases.append({"scenario": sc_full, "ctx": f"ctx/{model}_{sc_full}.json",
                              "golden": f"golden/{model}_{sc_full}.golden"})

        manifest["models"].append({"name": model, "tpl_file": tpl_file, "cases": cases})

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    n_total = sum(len(m["cases"]) for m in manifest["models"])
    print(f"scenarios: {scenarios}")
    print(f"models with template: {len(manifest['models'])}")
    for m in manifest["models"]:
        print(f"  {m['name']}: {len(m['cases'])} cases")
    print(f"TOTAL cases: {n_total}")
    print(f"output dir: {args.out}")


if __name__ == "__main__":
    main()

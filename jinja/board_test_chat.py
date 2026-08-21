#!/usr/bin/env python3
"""board_test_chat.py — 板端 chat_template 实时加载渲染测试

原理（模拟真实使用）：
  板端有模型目录（如 models/rknn3_zoo_test_*/）里的配置文件
  （chat_template.jinja / tokenizer_config.json）。测试即验证：板端从模型目录
  **实时加载模板并渲染** —— render_driver --model-dir <模型目录> --ctx <对话>。
  若没有模板配置文件，板端无法渲染；可推送data/models/下的模板配置文件到板端，模拟真实部署。

  因此：
    - 推送到板端的是「模型目录的模板配置文件 + 测试对话 ctx」，
      不推送旁路提取的模板文件（与真实部署资源一致）；
    - ctx 是测试输入（模拟对话），golden 是 PC 端 Python 生成的标准答案，仅用于对比。

测试方法论：PC 生成数据/golden → 推送板端 → 板端跑 C++ → 收回对比。

默认：板端模式（adb 连接，RK3588/RK3576）。
  --local 仅用于本机模拟验证（显式指定，非默认）。

用法：
  python board_test_chat.py --driver <render_driver路径或文件名> [--only gemma4,...] [--bench N]
  python board_test_chat.py --local --driver <本机 render_driver>     # 显式本机模拟

报告：data/board_test_report.md（含测试时间；渲染耗时 ms、峰值内存 MB）
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BOARD_MODELS = "/data/local/tmp/models"
DEFAULT_BOARD_DIR = "/data/local/tmp/chat_template_test"
BENCH = 5000  # 保证进程存活足够长，/proc 采样能捕捉峰值内存


def _host_models_dir(data_dir):
    return os.path.join(os.path.dirname(data_dir), "data", "models")


def parse_stderr(text):
    """从 stderr 解析 [bench] avg us 与 [mem] ru_maxrss KB；返回 (bench_ms, rss_kb)"""
    bench_ms = None
    rss_kb = None
    m = re.search(r"avg ([\d.]+) us", text)
    if m:
        bench_ms = float(m.group(1)) / 1000.0  # us -> ms
    m = re.search(r"ru_maxrss=(\d+) KB", text)
    if m:
        rss_kb = int(m.group(1))
    return bench_ms, rss_kb


class Board:
    """板端（adb）访问。峰值内存方法：后台跑进程 + 采样 /proc/$pid/status VmRSS。"""

    def __init__(self, board_dir, models_dir):
        self.board_dir = board_dir
        self.models_dir = models_dir

    def shell(self, cmd, timeout=300):
        r = subprocess.run(["adb", "shell", cmd], capture_output=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode

    def push(self, local, remote):
        r = subprocess.run(["adb", "push", local, remote], capture_output=True, text=True, timeout=180)
        return r.returncode, (r.stdout + r.stderr).strip()

    def run_driver(self, driver_rel, model_dir_abs, ctx_abs, bench):
        #   后台启动进程，循环采样 /proc/$pid/status 的 VmRSS，取最大值
        out_f = f"{self.board_dir}/_out.txt"
        err_f = f"{self.board_dir}/_err.txt"
        rss_f = f"{self.board_dir}/_rss.txt"
        wrapper = (
            f"rm -f {rss_f} {out_f} {err_f}; "
            f"({self.board_dir}/{driver_rel} --model-dir {model_dir_abs} --ctx {ctx_abs} --bench {bench} > {out_f} 2> {err_f}) & "
            f"pid=$!; peak=0; "
            f"while [ -d /proc/$pid ] 2>/dev/null; do "
            f"  rss=$(awk '/VmRSS/{{print $2}}' /proc/$pid/status 2>/dev/null); "
            f"  [ -n \"$rss\" ] && [ \"$rss\" -gt \"$peak\" ] && peak=$rss; "
            f"  sleep 0.01; "
            f"done; wait $pid; echo $peak > {rss_f}"
        )
        self.shell(wrapper)
        out, _, _ = self.shell(f"cat {out_f}")
        err, _, _ = self.shell(f"cat {err_f}")
        rss, _, _ = self.shell(f"cat {rss_f}")
        peak_kb = int(rss.strip()) if rss.strip().isdigit() else 0
        return 0, out, err, peak_kb


class LocalRunner:
    """本机模拟（显式 --local；不默认）。直接跑本机 render_driver，内存用其 --mem 自报 ru_maxrss。"""

    def __init__(self, driver, models_dir):
        self.driver = driver
        self.models_dir = models_dir
        self.is_windows = os.name == "nt"
        # Windows 本地模拟如需 cygwin 路径转换，显式设置环境变量 CYGWIN_BIN；默认不包装，直接执行
        self.cygwin_bin = os.environ.get("CYGWIN_BIN")

    def _cyg(self, p):
        r = subprocess.run([os.path.join(self.cygwin_bin, "cygpath.exe"), "-u", p],
                           capture_output=True, text=True)
        return r.stdout.strip()

    def run_driver(self, driver_rel, model_dir_abs, ctx_abs, bench):
        if self.is_windows and self.cygwin_bin:  # Windows 且显式设置 CYGWIN_BIN 时才用 cygwin bash 包装
            d, m, c = self._cyg(self.driver), self._cyg(model_dir_abs), self._cyg(ctx_abs)
            cmd = f"{d} --model-dir {m} --ctx {c} --bench {bench} --mem"
            r = subprocess.run([os.path.join(self.cygwin_bin, "bash.exe"), "-lc", cmd],
                               capture_output=True, timeout=180)
        else:
            driver = self.driver
            if self.is_windows and driver.lower().endswith(".exe") and os.path.exists(driver + ".exe"):
                driver += ".exe"
            r = subprocess.run([driver, "--model-dir", model_dir_abs, "--ctx", ctx_abs,
                                "--bench", str(bench), "--mem"],
                               capture_output=True, timeout=180)
        return r.returncode, r.stdout, r.stderr, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", required=True,
                    help="install 目录（如 install/chat_template_linux_aarch64），推送整个文件夹 demo/lib/include 到板端")
    ap.add_argument("--local", action="store_true", help="本机模拟（显式指定；默认是板端模式）")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--board-dir", default=DEFAULT_BOARD_DIR)
    ap.add_argument("--bench", type=int, default=BENCH)
    ap.add_argument("--only", default="", help="逗号分隔的模型名子串过滤")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data, "manifest.json"), encoding="utf-8"))
    models = manifest["models"]
    if args.only:
        subs = [s.strip() for s in args.only.split(",") if s.strip()]
        models = [m for m in models if any(s in m["name"] for s in subs)]

    host_models_dir = _host_models_dir(args.data)  # 本机 tokenizer/models
    install_dir = args.install
    driver_path = os.path.join(install_dir, "demo", "render_driver")

    if args.local:
        runner = LocalRunner(driver_path, host_models_dir)
        runner_name = "local（本机模拟）"
        print(f"[RUNNER] local  模型目录: {host_models_dir}")
    else:
        runner = Board(args.board_dir, BOARD_MODELS)
        runner_name = "board（adb 板端）"
        out, _, _ = runner.shell("echo OK", timeout=10)
        if b"OK" not in out:
            print("ERROR: 板端不可达（adb devices 检查）；如需本机模拟请显式加 --local")
            return 1
        # 推送整个 install 文件夹（demo + lib + include），模拟真实部署目录结构
        rc, msg = runner.push(driver_path, f"{args.board_dir}/render_driver")
        if rc != 0:
            print(f"ERROR: push render_driver 失败: {msg[:200]}"); return 1
        runner.shell(f"chmod +x {args.board_dir}/render_driver")
        for sub in ("lib", "include"):
            src = os.path.join(install_dir, sub)
            if os.path.isdir(src):
                runner.shell(f"mkdir -p {args.board_dir}/{sub}")
                for f in os.listdir(src):
                    rc, msg = runner.push(os.path.join(src, f), f"{args.board_dir}/{sub}/{f}")
                    if rc != 0:
                        print(f"WARN: push {sub}/{f} 失败: {msg[:150]}")
        print(f"[RUNNER] install 推送: {install_dir}（demo/lib/include）")
        # 推送各模型目录的模板配置文件（真实部署资源）+ 测试 ctx
        runner.shell(f"mkdir -p {args.board_dir}/ctx {runner.models_dir}")
        n_tpl = 0
        for m in models:
            src = os.path.join(host_models_dir, m["name"], m["tpl_file"])
            if not os.path.exists(src):
                print(f"WARN: 缺少模板配置文件 {src}，跳过 {m['name']}"); continue
            runner.shell(f"mkdir -p {runner.models_dir}/{m['name']}")
            rc, msg = runner.push(src, f"{runner.models_dir}/{m['name']}/{m['tpl_file']}")
            if rc != 0:
                print(f"WARN: push {m['name']} 模板配置失败: {msg[:150]}")
            n_tpl += 1
        rc, msg = runner.push(os.path.join(args.data, "ctx"), f"{args.board_dir}/")
        if rc != 0:
            print(f"WARN: push ctx 失败: {msg[:150]}")
        ctx_n, _, _ = runner.shell(f"ls {args.board_dir}/ctx | wc -l")
        print(f"[RUNNER] board via adb  模型配置推送数={n_tpl}  ctx文件数={ctx_n.strip()}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for mi, m in enumerate(models):
        model_dir_abs = f"{runner.models_dir}/{m['name']}"   # 板端模型目录（或本机模型目录）
        for case in m["cases"]:
            sc = case["scenario"]
            golden_path = os.path.join(args.data, case["golden"])
            # golden 缺失要显式标记 GOLDEN-MISS，不能静默当作 PASS
            if not os.path.exists(golden_path):
                golden = None
            else:
                golden = open(golden_path, "rb").read()
            ctx_abs = os.path.join(args.data, case["ctx"]) if args.local else f"{args.board_dir}/{case['ctx']}"
            # 长文本/长历史/长输入场景单次渲染较慢，bench 次数下调以免整体过慢
            slow_sc = sc.startswith(("longtext", "long_hist", "long_hist_long_input"))
            case_bench = min(args.bench, 20) if slow_sc else args.bench
            rc, out, err, peak_kb = runner.run_driver("render_driver", model_dir_abs, ctx_abs, case_bench)
            bench_ms, rss_kb = parse_stderr(err.decode("utf-8", "replace"))
            if peak_kb:                      # 板端 /proc 采样优先
                rss_kb = peak_kb
            mem_mb = rss_kb / 1024.0 if rss_kb else None

            if rc != 0:
                status, detail = "RENDER-ERR", err.decode("utf-8", "replace").strip()[:120]
            elif golden is None:
                status, detail = "GOLDEN-MISS", f"missing: {case['golden']}"
            elif out != golden:
                status, detail = "DIFF", f"cpp={len(out)}B golden={len(golden)}B"
            else:
                status, detail = "PASS", ""
            results.append({"model": m["name"], "scenario": sc, "status": status, "detail": detail,
                            "bench_ms": bench_ms, "mem_mb": mem_mb})
            b = f"{bench_ms:.3f}" if bench_ms is not None else "-"
            mem = f"{mem_mb:.2f}" if mem_mb is not None else "-"
            flag = "OK " if status == "PASS" else "XX "
            print(f"[{mi+1:>2}/{len(models)}] {m['name']:<32} {sc:<20} {flag}{status:<11} 耗时={b:>9}ms 峰值={mem:>6}MB {detail}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_total = len(results)
    print("=" * 78)
    print(f"PASS {n_pass}/{n_total}   FAIL {n_total - n_pass}")
    if n_pass == n_total and n_total:
        b_all = [r["bench_ms"] for r in results if r["bench_ms"]]
        m_all = [r["mem_mb"] for r in results if r["mem_mb"]]
        if b_all:
            print(f"render latency: min={min(b_all):.3f}ms  max={max(b_all):.3f}ms  avg={sum(b_all)/len(b_all):.3f}ms")
        if m_all:
            print(f"peak mem:       min={min(m_all):.2f}MB  max={max(m_all):.2f}MB  avg={sum(m_all)/len(m_all):.2f}MB")

    report_path = args.report or os.path.join(args.data, "board_test_report.md")
    _write_report(report_path, results, runner_name, n_pass, n_total, ts)
    print(f"report: {report_path}")
    return 0 if n_pass == n_total else 2


def _write_report(path, results, runner_name, n_pass, n_total, ts):
    lines = [
        "# chat_template 板端实时渲染测试报告",
        "",
        f"- 执行方式: `{runner_name}`",
        f"- 测试时间: {ts}",
        f"- 结果: **{n_pass}/{n_total}** 通过（与 Python golden 逐字节一致）",
        "",
        "| 模型 | 场景 | 状态 | 渲染耗时(ms) | 峰值内存(MB) | 备注 |",
        "|------|------|------|:----:|:----:|------|",
    ]
    for r in results:
        b = f"{r['bench_ms']:.3f}" if r["bench_ms"] is not None else "-"
        mem = f"{r['mem_mb']:.2f}" if r["mem_mb"] is not None else "-"
        lines.append(f"| {r['model']} | {r['scenario']} | {r['status']} | {b} | {mem} | {r['detail']} |")
    b_all = [r["bench_ms"] for r in results if r["bench_ms"]]
    m_all = [r["mem_mb"] for r in results if r["mem_mb"]]
    if b_all:
        lines += ["", "## 性能汇总", "",
                  f"- 渲染耗时（板端自测，ms）：min {min(b_all):.3f} / max {max(b_all):.3f} / avg {sum(b_all)/len(b_all):.3f}"]
    if m_all:
        lines += [f"- 峰值内存（/proc VmRSS 采样，MB）：min {min(m_all):.2f} / max {max(m_all):.2f} / avg {sum(m_all)/len(m_all):.2f}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())

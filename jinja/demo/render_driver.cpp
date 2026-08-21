// render_driver.cpp — RKNN3 chat_template 渲染驱动
// 只通过封装层 ChatTemplate 接口使用。
// 用法:
//   render_driver --model-dir <model_dir> --ctx <ctx.json> [--bench N] [--mem]
//     <model_dir> 为模型目录（含 chat_template.jinja 或 tokenizer_config.json），模拟板端实时加载
//   render_driver --tpl <chat_template.jinja | tokenizer_config.json> --ctx <ctx.json> [--bench N] [--mem]
//   render_driver --template "<inline template>" --ctx <ctx.json>
// 输出: 渲染后的 prompt 到 stdout；错误/统计到 stderr
#include "ChatTemplate.h"
#include "Log.h"

#include <nlohmann/json.hpp>

#include <sys/resource.h>
#include <sys/stat.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using json = nlohmann::json;

static bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// --tpl 指向 .jinja → 文件内容即模板；否则当 JSON 配置文件，取 chat_template 字段
static std::string load_template_source(const std::string& path, bool& ok) {
    ok = false;
    std::string content = read_file(path);
    if (content.empty()) { std::cerr << "cannot read: " << path << "\n"; return std::string(); }
    if (path.size() >= 6 && path.compare(path.size() - 6, 6, ".jinja") == 0) {
        ok = true;
        return content;
    }
    try {
        auto j = json::parse(content);
        if (j.contains("chat_template")) {
            if (j["chat_template"].is_string()) { ok = true; return j["chat_template"].get<std::string>(); }
            if (j["chat_template"].is_object() && j["chat_template"].contains("default")
                && j["chat_template"]["default"].is_string()) {
                ok = true;
                return j["chat_template"]["default"].get<std::string>();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "json parse error: " << e.what() << "\n";
    }
    return std::string();
}

int main(int argc, char** argv) {
    std::string tpl_path, tpl_str, ctx_path, model_dir;
    long bench = 0;
    bool do_mem = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--tpl" && i + 1 < argc) tpl_path = argv[++i];
        else if (a == "--template" && i + 1 < argc) tpl_str = argv[++i];
        else if (a == "--ctx" && i + 1 < argc) ctx_path = argv[++i];
        else if (a == "--bench" && i + 1 < argc) bench = std::atol(argv[++i]);
        else if (a == "--mem") do_mem = true;
        else if (a == "--log-level" && i + 1 < argc) {
            std::string lv = argv[++i];
            using chat_template::LogLevel;
            if (lv == "off") chat_template::SetLogLevel(LogLevel::Off);
            else if (lv == "error") chat_template::SetLogLevel(LogLevel::Error);
            else if (lv == "warn") chat_template::SetLogLevel(LogLevel::Warn);
            else if (lv == "info") chat_template::SetLogLevel(LogLevel::Info);
            else if (lv == "debug") chat_template::SetLogLevel(LogLevel::Debug);
            else { CT_LOGW("unknown --log-level '%s' (off/error/warn/info/debug)", lv.c_str()); }
        }
    }
    // Debug 构建默认 Debug；Release 构建默认 Warn。此处打印当前生效级别便于确认。
    CT_LOGI("log level = %d", static_cast<int>(chat_template::GetLogLevel()));

    // 通过封装层构造 ChatTemplate：model_dir 走封装层内部优先级（chat_template.jinja > tokenizer_config.json）
    std::unique_ptr<ChatTemplate> ct;
    if (!model_dir.empty()) {
        ct.reset(new ChatTemplate(model_dir.c_str()));
    } else if (!tpl_path.empty()) {
        bool ok = false;
        std::string src = load_template_source(tpl_path, ok);
        if (ok) ct.reset(new ChatTemplate(src.data(), src.size()));
    } else if (!tpl_str.empty()) {
        ct.reset(new ChatTemplate(tpl_str.data(), tpl_str.size()));
    }
    if (!ct || !ct->IsLoaded()) {
        CT_LOGE("template source not provided, empty, or load failed");
        std::cerr << "ERROR: template source not provided, empty, or load failed\n";
        return 2;
    }

    std::string ctx_json = "{}";   // 无 --ctx 时按空上下文渲染纯模板
    if (!ctx_path.empty())
        ctx_json = read_file(ctx_path);
    CT_LOGI("ctx: %zu bytes from %s", ctx_json.size(), ctx_path.empty() ? "<empty>" : ctx_path.c_str());

    std::string out;
    if (!ct->Render(ctx_json.data(), ctx_json.size(), &out)) {
        CT_LOGE("render failed");
        std::cerr << "ERROR: render failed\n";
        return 5;
    }
    CT_LOGD("render ok, %zu bytes output", out.size());
    std::cout << out;

    if (bench > 0) {
        auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < bench; ++i) {
            std::string tmp;
            if (!ct->Render(ctx_json.data(), ctx_json.size(), &tmp)) {
                CT_LOGE("render failed at iter %ld", i);
                std::cerr << "ERROR: render failed at iter " << i << "\n";
                return 5;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / bench;
        std::cerr << "[bench] " << bench << " iters, avg " << us << " us\n";
        CT_LOGD("bench %ld iters, avg %.2f us", bench, us);
    }
    if (do_mem) {
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
            std::cerr << "[mem] ru_maxrss=" << ru.ru_maxrss << " KB\n";
    }
    return 0;
}

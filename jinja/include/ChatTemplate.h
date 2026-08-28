#pragma once
#include <cstddef>
#include <memory>
#include <string>

class ChatTemplateImpl;

// RKNN3 chat_template 渲染封装层（基于 Jinja2Cpp）
//
// 仿 tokenizer 项目的 Tokenizer：demo / 板端程序只通过本接口使用，
// 不直接依赖 Jinja2Cpp / nlohmann 内部 API。链接 libchat_template.a 即可。
//
// 模板加载优先级（与 transformers 一致）：
//   从模型目录加载时，优先读取 <model_dir>/chat_template.jinja；
//   若文件不存在，则读 <model_dir>/tokenizer_config.json 的 chat_template 字段
//   （支持字符串形式，或 {"default": "..."} 对象形式）。
class ChatTemplate {
public:
    // 从 HF 模型目录加载模板（见类注释的优先级）
    explicit ChatTemplate(const char* model_dir);

    // 从模板源码字符串加载（UTF-8；末尾换行对齐 Python keep_trailing_newline=False 自动处理）
    ChatTemplate(const char* tpl, size_t tpl_len);

    ~ChatTemplate();

    ChatTemplate(const ChatTemplate&) = delete;
    ChatTemplate& operator=(const ChatTemplate&) = delete;

    bool IsLoaded() const;

    // 渲染对话上下文，输出渲染后的 prompt。
    //   ctx_json 形如：
    //     {"messages":[{"role":"user","content":"..."}, ...],
    //      "add_generation_prompt":true, "tools":[...], ...}
    // 成功返回 true 并写入 out；JSON 解析或模板渲染失败返回 false。
    bool Render(const char* ctx_json, size_t ctx_len, std::string* out);

    // 便捷重载
    bool Render(const std::string& ctx_json, std::string* out);

private:
    std::unique_ptr<ChatTemplateImpl> impl_;
};

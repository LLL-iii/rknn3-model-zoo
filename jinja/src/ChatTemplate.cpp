#include "ChatTemplate.h"
#include "Log.h"

#include <jinja2cpp/binding/nlohmann_json.h>
#include <jinja2cpp/template.h>
#include <nlohmann/json.hpp>

#include <sys/stat.h>

#include <fstream>
#include <sstream>

namespace
{

using json = nlohmann::json;

bool file_exists(const std::string& p)
{
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

std::string read_file(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f)
        return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// 从 tokenizer_config.json 内容提取 chat_template（字符串或 {"default": ...}）
std::string extract_chat_template(const std::string& content)
{
    try
    {
        auto j = json::parse(content);
        if (!j.contains("chat_template"))
            return std::string();
        const auto& ct = j["chat_template"];
        if (ct.is_string())
            return ct.get<std::string>();
        if (ct.is_object() && ct.contains("default") && ct["default"].is_string())
            return ct["default"].get<std::string>();
    }
    catch (...)
    {
    }
    return std::string();
}

} // namespace

class ChatTemplateImpl
{
public:
    bool loaded = false;
    jinja2::Template tpl;
};

ChatTemplate::ChatTemplate(const char* model_dir)
    : impl_(new ChatTemplateImpl)
{
    if (!model_dir)
        return;

    std::string src;
    std::string dir(model_dir);
    std::string jinja = dir + "/chat_template.jinja";
    if (file_exists(jinja))
    {
        src = read_file(jinja);
        CT_LOGI("load template from %s/chat_template.jinja (%zu bytes)", dir.c_str(), src.size());
    }
    else
    {
        src = extract_chat_template(read_file(dir + "/tokenizer_config.json"));
        CT_LOGI("load template from %s/tokenizer_config.json (chat_template field, %zu bytes)",
                dir.c_str(), src.size());
    }

    if (src.empty())
    {
        CT_LOGE("no template found in model dir: %s", dir.c_str());
        return;
    }
    // 对齐 Python jinja2 keep_trailing_newline=False
    if (src.back() == '\n')
        src.pop_back();

    impl_->loaded = static_cast<bool>(impl_->tpl.Load(src));
    CT_LOGI("template %s", impl_->loaded ? "loaded" : "load FAILED");
}

ChatTemplate::ChatTemplate(const char* tpl, size_t tpl_len)
    : impl_(new ChatTemplateImpl)
{
    if (!tpl)
    {
        CT_LOGE("ChatTemplate: null template pointer");
        return;
    }
    std::string src(tpl, tpl_len);
    if (!src.empty() && src.back() == '\n')
        src.pop_back();

    auto loadRes = impl_->tpl.Load(src);
    impl_->loaded = static_cast<bool>(loadRes);
    if (impl_->loaded)
        CT_LOGI("template loaded (%zu bytes)", src.size());
    else
        CT_LOGE("template load FAILED: %s", loadRes.error().ToString().c_str());
}

ChatTemplate::~ChatTemplate() = default;

bool ChatTemplate::IsLoaded() const
{
    return impl_->loaded;
}

bool ChatTemplate::Render(const char* ctx_json, size_t ctx_len, std::string* out)
{
    if (!impl_->loaded || !ctx_json || !out)
    {
        CT_LOGE("Render: not loaded or invalid args (loaded=%d)", impl_->loaded);
        return false;
    }
    try
    {
        auto j = json::parse(ctx_json, ctx_json + ctx_len);
        CT_LOGD("Render: ctx parsed, %zu keys", j.size());
        jinja2::ValuesMap params;
        for (auto it = j.begin(); it != j.end(); ++it)
            params[it.key()] = jinja2::Reflect(it.value());

        auto r = impl_->tpl.RenderAsString(params);
        if (!r)
        {
            CT_LOGE("Render: template render error: %s", r.error().ToString().c_str());
            return false;
        }
        *out = r.value();
        CT_LOGD("Render: %zu bytes output", out->size());
        return true;
    }
    catch (const std::exception& e)
    {
        CT_LOGE("Render: exception: %s", e.what());
        return false;
    }
}

bool ChatTemplate::Render(const std::string& ctx_json, std::string* out)
{
    return Render(ctx_json.data(), ctx_json.size(), out);
}

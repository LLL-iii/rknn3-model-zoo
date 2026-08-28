#include "filters.h"
#include "generic_adapters.h"
#include "out_stream.h"
#include "testers.h"
#include "value_helpers.h"
#include "value_visitors.h"

#include <fmt/args.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#ifdef JINJA2CPP_WITH_JSON_BINDINGS_BOOST
#include "binding/boost_json_serializer.h"
#elif JINJA2CPP_WITH_JSON_BINDINGS_NLOHMANN
#include "binding/nlohmann_json_serializer.h"
#else
#include "binding/rapid_json_serializer.h"
#endif



namespace jinja2
{
namespace filters
{
#ifndef JINJA2CPP_CHAT_ONLY
struct PrettyPrinter : visitors::BaseVisitor<std::string>
{
    using BaseVisitor::operator();

    PrettyPrinter(const RenderContext* context)
        : m_context(context)
    {
    }

    std::string operator()(const ListAdapter& list) const
    {
        std::string str;
        auto os = std::back_inserter(str);

        fmt::format_to(os, "[");
        bool isFirst = true;

        for (auto& v : list)
        {
            if (isFirst)
                isFirst = false;
            else
                fmt::format_to(os, ", ");
            fmt::format_to(os, "{}", Apply<PrettyPrinter>(v, m_context));
        }
        fmt::format_to(os, "]");

        return str;
    }

    std::string operator()(const MapAdapter& map) const
    {
        std::string str;
        auto os = std::back_inserter(str);

        fmt::format_to(os, "{{");

        const auto& keys = map.GetKeys();

        bool isFirst = true;
        for (auto& k : keys)
        {
            if (isFirst)
                isFirst = false;
            else
                fmt::format_to(os, ", ");

            fmt::format_to(os, "'{}': ", k);
            fmt::format_to(os, "{}", Apply<PrettyPrinter>(map.GetValueByName(k), m_context));
        }

        fmt::format_to(os, "}}");

        return str;
    }

    std::string operator()(const KeyValuePair& kwPair) const
    {
        std::string str;
        auto os = std::back_inserter(str);

        fmt::format_to(os, "'{}': ", kwPair.key);
        fmt::format_to(os, "{}", Apply<PrettyPrinter>(kwPair.value, m_context));

        return str;
    }

    std::string operator()(const std::string& str) const { return fmt::format("'{}'", str); }

    std::string operator()(const nonstd::string_view& str) const { return fmt::format("'{}'", fmt::basic_string_view<char>(str.data(), str.size())); }

    std::string operator()(const std::wstring& str) const { return fmt::format("'{}'", ConvertString<std::string>(str)); }

    std::string operator()(const nonstd::wstring_view& str) const { return fmt::format("'{}'", ConvertString<std::string>(str)); }

    std::string operator()(bool val) const { return val ? std::string("true") : std::string("false"); }

    std::string operator()(EmptyValue) const { return std::string("none"); }

    std::string operator()(const Callable&) const { return std::string("<callable>"); }

    std::string operator()(double val) const
    {
        std::string str;
        auto os = std::back_inserter(str);

        fmt::format_to(os, "{:.8g}", val);

        return str;
    }

    std::string operator()(int64_t val) const { return fmt::format("{}", val); }

    const RenderContext* m_context;
};
#endif

#ifndef JINJA2CPP_CHAT_ONLY
PrettyPrint::PrettyPrint(FilterParams params) {}

InternalValue PrettyPrint::Filter(const InternalValue& baseVal, RenderContext& context)
{
    return Apply<PrettyPrinter>(baseVal, &context);
}
#endif

Serialize::Serialize(const FilterParams params, const Serialize::Mode mode)
    : m_mode(mode)
{
    switch (mode)
    {
        case JsonMode:
            ParseParams({ { "indent", false, static_cast<int64_t>(0) } }, params);
            break;
        default:
            break;
    }
}

InternalValue Serialize::Filter(const InternalValue& value, RenderContext& context)
{
    if (m_mode == JsonMode)
    {
        const auto indent = ConvertToInt(this->GetArgumentValue("indent", context));
        return ToJson(value, indent);
    }

    return InternalValue();
}

namespace
{
#ifndef JINJA2CPP_CHAT_ONLY
using FormatContext = fmt::format_context;
using FormatArgument = fmt::basic_format_arg<FormatContext>;
using FormatDynamicArgsStore = fmt::dynamic_format_arg_store<FormatContext>;

struct FormatArgumentConverter : visitors::BaseVisitor<FormatArgument>
{
    using result_t = FormatArgument;

    using BaseVisitor::operator();

    FormatArgumentConverter(const RenderContext* context, FormatDynamicArgsStore& store)
        : m_context(context)
        , m_store(store)
    {
    }

    FormatArgumentConverter(const RenderContext* context, FormatDynamicArgsStore& store, const std::string& name)
        : m_context(context)
        , m_store(store)
        , m_name(name)
        , m_named(true)
    {
    }

    result_t operator()(const ListAdapter& list) const { return make_result(Apply<PrettyPrinter>(list, m_context)); }

    result_t operator()(const MapAdapter& map) const { return make_result(Apply<PrettyPrinter>(map, m_context)); }

    result_t operator()(const std::string& str) const { return make_result(str); }

    result_t operator()(const nonstd::string_view& str) const { return make_result(std::string(str.data(), str.size())); }

    result_t operator()(const std::wstring& str) const { return make_result(ConvertString<std::string>(str)); }

    result_t operator()(const nonstd::wstring_view& str) const { return make_result(ConvertString<std::string>(str)); }

    result_t operator()(double val) const { return make_result(val); }

    result_t operator()(int64_t val) const { return make_result(val); }

    result_t operator()(bool val) const { return make_result(val ? std::string("true") : std::string("false")); }

    result_t operator()(EmptyValue) const { return make_result(std::string("none")); }

    result_t operator()(const Callable&) const { return make_result(std::string("<callable>")); }

    template<typename T>
    result_t make_result(const T& t) const
    {
        if (!m_named)
        {
            m_store.push_back(t);
        }
        else
        {
            m_store.push_back(fmt::arg(m_name.c_str(), t));
        }
        // RKNN3 local patch: fmt's basic_format_arg(T) ctor is private since fmt 6+;
        // use the public friend helper fmt::detail::make_arg instead.
        return fmt::detail::make_arg<FormatContext>(t);
    }

    const RenderContext* m_context{};
    FormatDynamicArgsStore& m_store;
    const std::string m_name;
    bool m_named = false;
};
#endif

} // namespace

#ifndef JINJA2CPP_CHAT_ONLY
InternalValue StringFormat::Filter(const InternalValue& baseVal, RenderContext& context)
{
    // Format library internally likes using non-owning views to complex arguments.
    // In order to ensure proper lifetime of values and named args,
    // helper buffer is created and passed to visitors.
    FormatDynamicArgsStore store;
    for (auto& arg : m_params.posParams)
    {
        Apply<FormatArgumentConverter>(arg->Evaluate(context), &context, store);
    }

    for (auto& arg : m_params.kwParams)
    {
        Apply<FormatArgumentConverter>(arg.second->Evaluate(context), &context, store, arg.first);
    }

    return InternalValue(fmt::vformat(AsString(baseVal), store));
}
#endif

#ifndef JINJA2CPP_CHAT_ONLY
class XmlAttrPrinter : public visitors::BaseVisitor<std::string>
{
public:
    using BaseVisitor::operator();

    explicit XmlAttrPrinter(RenderContext* context, bool isFirstLevel = false)
        : m_context(context)
        , m_isFirstLevel(isFirstLevel)
    {
    }

    std::string operator()(const ListAdapter& list) const
    {
        EnforceThatNested();

        return EscapeHtml(Apply<PrettyPrinter>(list, m_context));
    }

    std::string operator()(const MapAdapter& map) const
    {
        if (!m_isFirstLevel)
        {
            return EscapeHtml(Apply<PrettyPrinter>(map, m_context));
        }

        std::string str;
        auto os = std::back_inserter(str);

        const auto& keys = map.GetKeys();

        bool isFirst = true;
        for (auto& k : keys)
        {
            const auto& v = map.GetValueByName(k);
            const auto item = Apply<XmlAttrPrinter>(v, m_context, false);
            if (item.length() > 0)
            {
                if (isFirst)
                    isFirst = false;
                else
                    fmt::format_to(os, " ");

                fmt::format_to(os, "{}=\"{}\"", k, item);
            }
        }

        return str;
    }

    std::string operator()(const KeyValuePair& kwPair) const
    {
        EnforceThatNested();

        return EscapeHtml(Apply<PrettyPrinter>(kwPair, m_context));
    }

    std::string operator()(const std::string& str) const
    {
        EnforceThatNested();

        return EscapeHtml(str);
    }

    std::string operator()(const nonstd::string_view& str) const
    {
        EnforceThatNested();

        const auto result = fmt::format("{}", fmt::basic_string_view<char>(str.data(), str.size()));
        return EscapeHtml(result);
    }

    std::string operator()(const std::wstring& str) const
    {
        EnforceThatNested();

        return EscapeHtml(ConvertString<std::string>(str));
    }

    std::string operator()(const nonstd::wstring_view& str) const
    {
        EnforceThatNested();

        const auto result = fmt::format("{}", ConvertString<std::string>(str));
        return EscapeHtml(result);
    }

    std::string operator()(bool val) const
    {
        EnforceThatNested();

        return val ? std::string("true") : std::string("false");
    }

    std::string operator()(EmptyValue) const
    {
        EnforceThatNested();

        return std::string("");
    }

    std::string operator()(const Callable&) const
    {
        EnforceThatNested();

        return std::string("");
    }

    std::string operator()(double val) const
    {
        EnforceThatNested();

        std::string str;
        auto os = std::back_inserter(str);

        fmt::format_to(os, "{:.8g}", val);

        return str;
    }

    std::string operator()(int64_t val) const
    {
        EnforceThatNested();

        return fmt::format("{}", val);
    }

private:
    void EnforceThatNested() const
    {
        if (m_isFirstLevel)
            m_context->GetRendererCallback()->ThrowRuntimeError(ErrorCode::InvalidValueType, ValuesList{});
    }

    std::string EscapeHtml(const std::string &str) const
    {
        const auto result = std::accumulate(str.begin(), str.end(), std::string(""), [](const std::string& str, const char c)
        {
            switch (c)
            {
            case '<':
                return str + "&lt;";
                break;
            case '>':
                return str +"&gt;";
                break;
            case '&':
                return str +"&amp;";
                break;
            case '\'':
                return str +"&#39;";
                break;
            case '\"':
                return str +"&#34;";
                break;
            default:
                return str + c;
                break;
            }
        });

        return result;
    }

private:
    RenderContext* m_context{};
    bool m_isFirstLevel{};
};
#endif

#ifndef JINJA2CPP_CHAT_ONLY
XmlAttrFilter::XmlAttrFilter(FilterParams) {}

InternalValue XmlAttrFilter::Filter(const InternalValue& baseVal, RenderContext& context)
{
    return Apply<XmlAttrPrinter>(baseVal, &context, true);
}
#endif

} // namespace filters
} // namespace jinja2

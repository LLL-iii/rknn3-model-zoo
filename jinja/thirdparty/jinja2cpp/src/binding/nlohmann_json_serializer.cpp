#include "nlohmann_json_serializer.h"

#include "../value_visitors.h"

#include <fmt/ostream.h>

#include <iterator>
#include <numeric>
#include <string>


namespace jinja2
{
namespace nlohmann_json_serializer
{
namespace
{
struct JsonInserter : visitors::BaseVisitor<nlohmann::json>
{
    using BaseVisitor::operator();

    explicit JsonInserter() {}

    nlohmann::json operator()(const ListAdapter& list) const
    {
        nlohmann::json listValue = nlohmann::json::array();
        for (auto& v : list)
        {
            listValue.push_back(Apply<JsonInserter>(v));
        }
        return listValue;
    }

    nlohmann::json operator()(const MapAdapter& map) const
    {
        nlohmann::json mapNode = nlohmann::json::object();
        const auto& keys = map.GetKeys();
        for (auto& k : keys)
        {
            mapNode.emplace(k.c_str(), Apply<JsonInserter>(map.GetValueByName(k)));
        }

        return mapNode;
    }

    nlohmann::json operator()(const KeyValuePair& kwPair) const
    {
        nlohmann::json pairNode = nlohmann::json::object();
        pairNode.emplace(kwPair.key.c_str(), Apply<JsonInserter>(kwPair.value));
        return pairNode;
    }

    nlohmann::json operator()(const std::string& str) const
    {
        return nlohmann::json(str);
    }

    nlohmann::json operator()(const nonstd::string_view& str) const
    {
        return nlohmann::json(str);
    }

    nlohmann::json operator()(const std::wstring& str) const
    {
        auto s = ConvertString<std::string>(str);
        return nlohmann::json(s);
    }

    nlohmann::json operator()(const nonstd::wstring_view& str) const
    {
        auto s = ConvertString<std::string>(str);
        return nlohmann::json(s);
    }

    nlohmann::json operator()(bool val) const { return nlohmann::json(val); }

    nlohmann::json operator()(EmptyValue) const { return nlohmann::json(); }

    nlohmann::json operator()(const Callable&) const { return nlohmann::json("<callable>"); }

    nlohmann::json operator()(double val) const { return nlohmann::json(val); }

    nlohmann::json operator()(int64_t val) const { return nlohmann::json(val); }
};
// 与 Python json.dumps(sort_keys=True) 对齐：对象/数组使用 ", " 与 ": " 分隔，
// 键按 nlohmann object（std::map）字母序；标量由 nlohmann dump 负责（ensure_ascii 对齐 Python）。
std::string DumpSpaced(const nlohmann::json& j)
{
    if (j.is_object())
    {
        std::string out = "{";
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            if (!first)
                out += ", ";
            first = false;
            out += nlohmann::json(it.key()).dump(-1, ' ', true);
            out += ": ";
            out += DumpSpaced(it.value());
        }
        return out + "}";
    }
    if (j.is_array())
    {
        std::string out = "[";
        bool first = true;
        for (const auto& el : j)
        {
            if (!first)
                out += ", ";
            first = false;
            out += DumpSpaced(el);
        }
        return out + "]";
    }
    return j.dump(-1, ' ', true);
}
} // namespace
} // namespace nlohmann_json_serializer

std::string ToJson(const InternalValue& value, uint8_t indent)
{
    auto jsonValue = Apply<nlohmann_json_serializer::JsonInserter>(value);
    auto jsonString = indent == 0 ? nlohmann_json_serializer::DumpSpaced(jsonValue) : jsonValue.dump(indent);
    const auto result = std::accumulate(jsonString.begin(), jsonString.end(), std::string(), [](const std::string& str, const char c)
    {
        switch (c)
        {
       case '<':
            return str + "\\u003c";
            break;
        case '>':
            return str +"\\u003e";
            break;
        case '&':
            return str +"\\u0026";
            break;
        case '\'':
            return str +"\\u0027";
            break;
        default:
            return str + c;
            break;
        }
    });
    return result;
}

} // namespace jinja2

#ifndef JINJA2CPP_STRING_LITERALS_COMPAT_H
#define JINJA2CPP_STRING_LITERALS_COMPAT_H

// C++11 兼容头：原提供 operator""s（C++14 标准库才有）。
// 代码中所有 "..."s 字面量已替换为 std::string(...)，此头现为 no-op。
// 注意：Clang 在 C++11 下将非下划线后缀 operator""s 直接视为错误
// （-Wreserved-user-defined-literal），故不再定义该运算符。

#endif // JINJA2CPP_STRING_LITERALS_COMPAT_H

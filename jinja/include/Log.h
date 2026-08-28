#pragma once
#include <cstdio>

// 日志（仿 tokenizer 的 src/utils/Log.h，但输出到 stderr 以免污染渲染 stdout）
// 级别划分：
//   Release 构建（NDEBUG）：默认 Warn（只输出错误/警告）
//   Debug 构建（非 NDEBUG）：默认 Debug（输出全部）
// 运行期可用 SetLogLevel 覆盖默认。
namespace chat_template {

enum class LogLevel {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
};

LogLevel DefaultLogLevel();
void SetLogLevel(LogLevel lv);
LogLevel GetLogLevel();

} // namespace chat_template

#define CT_LOG(lv, fmt, ...) do {                                                 \
    if (static_cast<int>(lv) <= static_cast<int>(chat_template::GetLogLevel()))   \
        std::fprintf(stderr, "[CT][%c] %s:%d " fmt "\n",                          \
                     "EWID"[static_cast<int>(lv) - 1], __FUNCTION__, __LINE__,    \
                     ##__VA_ARGS__);                                              \
} while (0)

#define CT_LOGE(...) CT_LOG(chat_template::LogLevel::Error, __VA_ARGS__)
#define CT_LOGW(...) CT_LOG(chat_template::LogLevel::Warn,  __VA_ARGS__)
#define CT_LOGI(...) CT_LOG(chat_template::LogLevel::Info,  __VA_ARGS__)
#define CT_LOGD(...) CT_LOG(chat_template::LogLevel::Debug, __VA_ARGS__)

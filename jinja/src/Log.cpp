#include "Log.h"

namespace chat_template {

LogLevel DefaultLogLevel()
{
#ifdef NDEBUG
    return LogLevel::Warn; // Release：只输出错误/警告
#else
    return LogLevel::Debug; // Debug：输出全部
#endif
}

static LogLevel g_logLevel = DefaultLogLevel();

void SetLogLevel(LogLevel lv)
{
    g_logLevel = lv;
}

LogLevel GetLogLevel()
{
    return g_logLevel;
}

} // namespace chat_template

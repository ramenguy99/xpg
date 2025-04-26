#pragma once

#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <atomic>

// NOTE: this will only do checking in msvc with versions that support /analyze
#ifdef _MSC_VER
#include <stddef.h>
#ifdef _USE_ATTRIBUTES_FOR_SAL
#undef _USE_ATTRIBUTES_FOR_SAL
#endif
/* nolint */
#define _USE_ATTRIBUTES_FOR_SAL 1
#include <sal.h> // @manual
#define PRINTF_FORMAT _Printf_format_string_
#define PRINTF_FORMAT_ATTR(format_param, dots_param) /**/
#else
#define PRINTF_FORMAT /**/
#define PRINTF_FORMAT_ATTR(format_param, dots_param) \
  __attribute__((__format__(__printf__, format_param, dots_param)))
#endif

#include "defines.h"
#include "platform.h"

namespace xpg {
namespace logging {

enum class LogLevel : u32 {
    Trace = 0,
    Debug,
    Info,
    Warning,
    Error,
    Disabled,
};

inline const char* log_level_to_string[] = {
    "TRACE",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "DISABLED",
};

inline void log_stdout(LogLevel level, const char* ctx, const char* fmt, va_list args);
typedef void (*LogFunc)(LogLevel level, const char* ctx, const char* fmt, va_list args);

inline std::atomic<LogLevel> g_log_level = LogLevel::Info;
inline std::atomic<LogFunc> g_log_func = log_stdout;

inline void set_log_level(LogLevel level) {
    g_log_level.store(level, std::memory_order_seq_cst);
}

inline void log_stdout(LogLevel level, const char* ctx, const char* fmt, va_list args) {
    platform::SystemTime system_time = {};
    platform::GetLocalTime(&system_time);

    // TODO: likely grab mutex here to prevent log mixing? (need a mutex with OS block in threading.h)
    const char* level_str = (u32)level < ArrayCount(log_level_to_string) ? log_level_to_string[(u32)level] : "";
    printf("[%04u-%02u-%02u %02u:%02u:%02u.%03u] %-6s [%s] ",
        system_time.year,
        system_time.month,
        system_time.day,
        system_time.hour,
        system_time.minute,
        system_time.second,
        system_time.milliseconds,
        level_str, ctx);
    vprintf(fmt, args);
    printf("\n");
}

#define DEFINE_LOG_FUNC(name, level) \
    PRINTF_FORMAT_ATTR(2,3) \
    inline void name(const char* ctx, PRINTF_FORMAT const char* fmt, ...) { \
        if (g_log_level.load(std::memory_order_relaxed) > LogLevel::level) { \
            return; \
        } \
        va_list args; \
        va_start(args, fmt); \
        LogFunc func = g_log_func.load(std::memory_order_relaxed); \
        if (func) { \
            func(LogLevel::level, ctx, fmt, args); \
        } \
        va_end(args); \
    }

DEFINE_LOG_FUNC(error,   Error)
DEFINE_LOG_FUNC(warning, Warning)
DEFINE_LOG_FUNC(debug, Debug)
DEFINE_LOG_FUNC(info, Info)
DEFINE_LOG_FUNC(trace, Trace)

#undef DEFINE_LOG_FUNC

} // namespace logging
} // namespace xpg
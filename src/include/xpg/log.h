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

namespace logging
{

enum class LogLevel : u32 {
    Trace = 0,
    Debug,
    Info,
    Warning,
    Error,
    Disabled,
};

inline std::atomic<LogLevel> g_log_level = LogLevel::Info;

inline void set_log_level(LogLevel level) {
    g_log_level.store(level, std::memory_order_seq_cst);
}

inline void log_internal(const char* level_string, const char* ctx, const char* fmt, va_list args) {
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[128];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "[%H:%M:%S %d-%m-%Y]", timeinfo);

    printf("%s %-6s [%s] ", buffer, level_string, ctx);
    vprintf(fmt, args);
    printf("\n");
}

#define DEFINE_LOG_FUNC(name, level, str) \
    PRINTF_FORMAT_ATTR(2,3) \
    inline void name(const char* ctx, PRINTF_FORMAT const char* fmt, ...) { \
        if (g_log_level.load(std::memory_order_relaxed) > LogLevel::level) { \
            return; \
        } \
        va_list args; \
        va_start(args, fmt); \
        log_internal(str, ctx, fmt, args); \
        va_end(args); \
    }

DEFINE_LOG_FUNC(error,   Error, "error")
DEFINE_LOG_FUNC(warning, Warning, "warning")
DEFINE_LOG_FUNC(debug, Debug, "debug")
DEFINE_LOG_FUNC(info, Info, "info")
DEFINE_LOG_FUNC(trace, Trace, "trace")

#undef DEFINE_LOG_FUNC

}

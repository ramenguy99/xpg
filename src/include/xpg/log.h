#pragma once

#include <stdio.h>
#include <time.h>
#include <stdarg.h>
#include <atomic>

#include "defines.h"

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

inline void error(const char* ctx, PRINTF_FORMAT const char* fmt, ...) {
    if (g_log_level.load(std::memory_order_relaxed) > LogLevel::Error) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    log_internal("error", ctx, fmt, args);
    va_end(args);
}

inline void warning(const char* ctx, PRINTF_FORMAT const char* fmt, ...) {
    if (g_log_level.load(std::memory_order_relaxed) > LogLevel::Warning) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    log_internal("warning", ctx, fmt, args);
    va_end(args);
}

inline void info(const char* ctx, PRINTF_FORMAT const char* fmt, ...) {
    if (g_log_level.load(std::memory_order_relaxed) > LogLevel::Info) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    log_internal("info", ctx, fmt, args);
    va_end(args);
}

}

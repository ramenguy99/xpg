// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

#include <atomic>
#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/intrusive/counter.inl>

#include <xpg/log.h>

namespace nb = nanobind;

void gfx_create_bindings(nb::module_&);
void imgui_create_bindings(nb::module_&);

#if PYXPG_SLANG_ENABLED
void slang_create_bindings(nb::module_&);
#endif

static void log_impl(xpg::logging::LogLevel level, const char* ctx, const char* fmt, va_list args) {
    // If Python has already shut down log to stdout normally.
    if (!nb::is_alive()) {
        xpg::logging::log_stdout(level, ctx, fmt, args);
        return;
    }

    // Must hold the GIL to access the functionality below
    nb::gil_scoped_acquire guard_gil;

    // Temporarily clear error status flags, if present
    nb::error_scope guard_error;

    // Get stdout
    nb::handle file = PySys_GetObject("stdout");

    // Print time, level and context
    char buf[1024];
    {
        xpg::platform::SystemTime system_time = {};
        xpg::platform::GetLocalTime(&system_time);
        const char* level_str = (u32)level < ArrayCount(xpg::logging::log_level_to_string) ? xpg::logging::log_level_to_string[(u32)level] : "";

        int n = snprintf(buf, sizeof(buf), "[%04u-%02u-%02u %02u:%02u:%02u.%03u] %-6s [%s] ",
            system_time.year,
            system_time.month,
            system_time.day,
            system_time.hour,
            system_time.minute,
            system_time.second,
            system_time.milliseconds,
            level_str,
            ctx
        );
        if (n < sizeof(buf)) {
            PyFile_WriteString(buf, file.ptr());
        } else {
            char* alloced_buf = (char*)malloc(n + 1);
            vsnprintf(alloced_buf, n + 1, fmt, args);
            PyFile_WriteString(alloced_buf, file.ptr());
            free(alloced_buf);
        }
    }

    // Print format string
    {
        int n = vsnprintf(buf, sizeof(buf), fmt, args);
        if (n < sizeof(buf)) {
            PyFile_WriteString(buf, file.ptr());
        } else {
            char* alloced_buf = (char*)malloc(n + 1);
            vsnprintf(alloced_buf, n + 1, fmt, args);
            PyFile_WriteString(alloced_buf, file.ptr());
            free(alloced_buf);
        }
    }

    PyFile_WriteString("\n", file.ptr());
}

void log_create_bindings(nb::module_ &m, PyModuleDef &pmd) {
    // Initialize logging
    xpg::logging::g_log_level = xpg::logging::LogLevel::Disabled;
    xpg::logging::g_log_func = log_impl;

    // Export log level controls
    nb::enum_<xpg::logging::LogLevel>(m, "LogLevel")
        .value("TRACE", xpg::logging::LogLevel::Trace)
        .value("DEBUG", xpg::logging::LogLevel::Debug)
        .value("INFO", xpg::logging::LogLevel::Info)
        .value("WARN", xpg::logging::LogLevel::Warning)
        .value("ERROR", xpg::logging::LogLevel::Error)
        .value("DISABLED", xpg::logging::LogLevel::Disabled)
    ;

    // m.def("set_log_level", [](xpg::logging::LogLevel&){}, nb::arg("level"));
    m.def("set_log_level", xpg::logging::set_log_level, nb::arg("level"));
    m.def("get_log_level", [](){ return xpg::logging::g_log_level.load(std::memory_order_relaxed); });

    pmd.m_free = [](void *) {
        // Switch from the Python logger to standard stderr output
        xpg::logging::g_log_func = xpg::logging::log_stdout;
    };
}

NB_MODULE(_pyxpg, m) {
    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        });

    log_create_bindings(m, nanobind__pyxpg_module);

    gfx_create_bindings(m);

    nb::module_ mod_imgui = m.def_submodule("imgui", "ImGui bindings for XPG");
    imgui_create_bindings(mod_imgui);

#if PYXPG_SLANG_ENABLED
    nb::module_ mod_slang = m.def_submodule("slang", "Slang bindings for XPG");
    slang_create_bindings(mod_slang);
#endif
}

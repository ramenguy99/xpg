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


std::function<void(xpg::logging::LogLevel, nb::str, nb::str)> g_log_callback;

void log_impl(xpg::logging::LogLevel level, const char* ctx, const char* fmt, va_list args) {
    nb::gil_scoped_acquire acq;
    try {
        if(g_log_callback) {
            char buf[1024];
            nb::str str;
            int n = vsnprintf(buf, sizeof(buf), fmt, args);
            if (n < sizeof(buf)) {
                str = nb::str(buf);
            } else {
                char* alloced_buf = (char*)malloc(n + 1);
                vsnprintf(alloced_buf, n + 1, fmt, args);
                str = nb::str(alloced_buf);
                free(alloced_buf);
            }
            g_log_callback(level, nb::str(ctx), str);
        }
    } catch (nb::python_error &e) {
        e.restore();
    }
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

    // Initialize logging
    xpg::logging::g_log_func = log_impl;

    nb::enum_<xpg::logging::LogLevel>(m, "LogLevel")
        .value("TRACE", xpg::logging::LogLevel::Trace)
        .value("DEBUG", xpg::logging::LogLevel::Debug)
        .value("INFO", xpg::logging::LogLevel::Info)
        .value("WARN", xpg::logging::LogLevel::Warning)
        .value("ERROR", xpg::logging::LogLevel::Error)
        .value("DISABLED", xpg::logging::LogLevel::Disabled)
    ;

    m.def("set_log_callback", [](std::optional<std::function<void(xpg::logging::LogLevel, nb::str, nb::str)>> callback) {
        g_log_callback = std::move(callback.value_or(nullptr));
    }, nb::arg("callback").none());


    gfx_create_bindings(m);

    nb::module_ mod_imgui = m.def_submodule("imgui", "ImGui bindings for XPG");
    imgui_create_bindings(mod_imgui);

#if PYXPG_SLANG_ENABLED
    nb::module_ mod_slang = m.def_submodule("slang", "Slang bindings for XPG");
    slang_create_bindings(mod_slang);
#endif
}

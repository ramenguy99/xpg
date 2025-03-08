#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/intrusive/counter.inl>

namespace nb = nanobind;

void gfx_create_bindings(nb::module_&);
void imgui_create_bindings(nb::module_&);
void slang_create_bindings(nb::module_&);

NB_MODULE(pyxpg, m) {
    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        });
    
    gfx_create_bindings(m);

    nb::module_ mod_imgui = m.def_submodule("imgui", "ImGui bindings for XPG");
    imgui_create_bindings(mod_imgui);

    nb::module_ mod_slang = m.def_submodule("slang", "Slang bindings for XPG");
    slang_create_bindings(mod_slang);
}

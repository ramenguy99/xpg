#include <array>
#include <vector>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <imgui.h>

namespace nb = nanobind;

typedef ImU32 Color;

struct DrawList: nb::intrusive_base {
    DrawList(ImDrawList* list): list(list) { }
    ImDrawList* list;
};

void imgui_create_bindings(nb::module_& mod_imgui)
{
    nb::class_<ImVec2>(mod_imgui, "Vec2")
        .def(nb::init<>())
        .def(nb::init<float, float>(), nb::arg("x"), nb::arg("y"))
        .def_rw("x", &ImVec2::x)
        .def_rw("y", &ImVec2::y)
        .def("__init__", [](ImVec2* v, nb::tuple t) {
            if (t.size() != 2) {
                nb::raise_type_error("Cannot convert tuple of length %zu to Vec2", t.size());
            }
            new (v) ImVec2(nb::cast<float>(t[0]), nb::cast<float>(t[1]));
            }, nb::arg("t"))
        .def("__init__", [](ImVec2* v, nb::list l) {
            if (l.size() != 2) {
                nb::raise_type_error("Cannot convert tuple of length %zu to Vec2", l.size());
            }
            new (v) ImVec2(nb::cast<float>(l[0]), nb::cast<float>(l[1]));
        }, nb::arg("l"))
    ;
    nb::implicitly_convertible<nb::tuple, ImVec2>();
    nb::implicitly_convertible<nb::list, ImVec2>();
    
    nb::class_<ImVec4>(mod_imgui, "Vec4")
        .def(nb::init<>())
        .def(nb::init<float, float, float, float>(), nb::arg("x"), nb::arg("y"), nb::arg("w"), nb::arg("w"))
        .def_rw("x", &ImVec4::x)
        .def_rw("y", &ImVec4::y)
        .def_rw("z", &ImVec4::z)
        .def_rw("w", &ImVec4::w)
        .def("__init__", [](ImVec2* v, nb::tuple t) {
            if (t.size() != 4) {
                nb::raise_type_error("Cannot convert tuple of length %zu to Vec4", t.size());
            }
            new (v) ImVec4(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]));
            }, nb::arg("t"))
        .def("__init__", [](ImVec2* v, nb::list l) {
            if (l.size() != 4) {
                nb::raise_type_error("Cannot convert tuple of length %zu to Vec2", l.size());
            }
            new (v) ImVec4(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]));
        }, nb::arg("l"))
    ;
    nb::implicitly_convertible<nb::tuple, ImVec4>();
    nb::implicitly_convertible<nb::list, ImVec4>();

    #include "generated_imgui.inc"
}

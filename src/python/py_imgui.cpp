#include <array>
#include <vector>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <imgui.h>

namespace nb = nanobind;

typedef ImU32 Color;

void imgui_create_bindings(nb::module_& mod_imgui)
{
    #include "generated_imgui.inc"
    nb::class_<ImVec2>(mod_imgui, "Vec2")
        .def(nb::init<>())
        .def(nb::init<float, float>())
        .def_rw("x", &ImVec2::x)
        .def_rw("y", &ImVec2::y)
    ;
    
    nb::class_<ImVec4>(mod_imgui, "Vec4")
        .def(nb::init<>())
        .def(nb::init<float, float, float, float>())
        .def_rw("x", &ImVec4::x)
        .def_rw("y", &ImVec4::y)
        .def_rw("z", &ImVec4::z)
        .def_rw("w", &ImVec4::w)
    ;
}

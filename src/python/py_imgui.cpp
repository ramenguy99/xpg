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
#include <nanobind/ndarray.h>

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
        .def(nb::init<float, float, float, float>(), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
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

    drawlist_class.def("add_rect_batch",
        [](DrawList& self,
           nb::ndarray<float, nb::shape<-1, 2>> p_min,
           nb::ndarray<float, nb::shape<-1, 2>> p_max,
           nb::ndarray<uint32_t, nb::shape<-1>> col,
           nb::ndarray<float, nb::shape<-1>> rounding,
           nb::ndarray<float, nb::shape<-1>> thickness
        ) {

            bool col_per_object = col.shape(0) != 1;
            bool rounding_per_object = rounding.shape(0) != 1;
            bool thickness_per_object = thickness.shape(0) != 1;
            for (size_t i = 0; i < p_min.shape(0); i++) {
                self.list->AddRect(
                    ImVec2(p_min(i, 0), p_min(i, 1)),
                    ImVec2(p_max(i, 0), p_max(i, 1)),
                    col_per_object ? col(i) : col(0),
                    rounding_per_object ? rounding(i) : rounding(0),
                    thickness_per_object ? thickness(i) : thickness(0)
                );
            }
        },
        nb::arg("p_min"),
        nb::arg("p_max"),
        nb::arg("color"),
        nb::arg("rounding"),
        nb::arg("thickness")
    );

    drawlist_class.def("add_rect_filled_batch",
        [](DrawList& self,
           nb::ndarray<float, nb::shape<-1, 2>> p_min,
           nb::ndarray<float, nb::shape<-1, 2>> p_max,
           nb::ndarray<uint32_t, nb::shape<-1>> col,
           nb::ndarray<float, nb::shape<-1>> rounding
        ) {

            bool col_per_object = col.shape(0) != 1;
            bool rounding_per_object = rounding.shape(0) != 1;
            for (size_t i = 0; i < p_min.shape(0); i++) {
                self.list->AddRectFilled(
                    ImVec2(p_min(i, 0), p_min(i, 1)),
                    ImVec2(p_max(i, 0), p_max(i, 1)),
                    col_per_object ? col(i) : col(0),
                    rounding_per_object ? rounding(i) : rounding(0)
                    );
            }
        },
        nb::arg("p_min"),
        nb::arg("p_max"),
        nb::arg("color"),
        nb::arg("rounding")
    );

    // IO
    mod_imgui.def("get_io", ImGui::GetIO, nb::rv_policy::reference);
}

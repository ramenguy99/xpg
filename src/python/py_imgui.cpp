// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

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
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <imgui.h>
#include <implot.h>

#include "py_gfx.h"


namespace nb = nanobind;

typedef ImU32 Color;

struct DrawList: nb::intrusive_base {
    DrawList(ImDrawList* list): list(list) { }
    ImDrawList* list;
};

struct Texture: nb::intrusive_base {
    Texture(nb::ref<DescriptorSet> descriptor_set)
        : descriptor_set(std::move(descriptor_set)) {
        tex_ref = ImTextureRef((ImTextureID)this->descriptor_set->set.set);
    }

    void destroy() {
        if (descriptor_set) {
            descriptor_set.reset();
        }
    }

    ~Texture() {
        destroy();
    }

    nb::ref<DescriptorSet> descriptor_set;
    ImTextureRef tex_ref;
};


static const char* dtype_code_to_str(u8 code) {
    switch(code) {
        case (int)nb::dlpack::dtype_code::Int: return "int";
        case (int)nb::dlpack::dtype_code::UInt: return "uint";
        case (int)nb::dlpack::dtype_code::Float: return "float";
        default: return "<unknown>";
    }
}

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
        .def("__repr__", [](ImVec2& v) {
            return nb::str("[{}, {}]").format(v.x, v.y);
        })
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
        .def("__repr__", [](ImVec4& v) {
            return nb::str("[{}, {}, {}, {}]").format(v.x, v.y, v.z, v.w);
        })
    ;
    nb::implicitly_convertible<nb::tuple, ImVec4>();
    nb::implicitly_convertible<nb::list, ImVec4>();

    #include "generated_imgui.inc"

    drawlist_class.def("add_line_batch",
        [](DrawList& self,
           nb::ndarray<float, nb::shape<-1, 2>> p1,
           nb::ndarray<float, nb::shape<-1, 2>> p2,
           nb::ndarray<uint32_t, nb::shape<-1>> col,
           nb::ndarray<float, nb::shape<-1>> thickness
        ) {

            bool col_per_object = col.shape(0) != 1;
            bool thickness_per_object = thickness.shape(0) != 1;
            for (size_t i = 0; i < p1.shape(0); i++) {
                self.list->AddLine(
                    ImVec2(p1(i, 0), p2(i, 1)),
                    ImVec2(p1(i, 0), p2(i, 1)),
                    col_per_object ? col(i) : col(0),
                    thickness_per_object ? thickness(i) : thickness(0)
                );
            }
        },
        nb::arg("p1"),
        nb::arg("p2"),
        nb::arg("color"),
        nb::arg("thickness")
    );

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

    drawlist_class.def("add_circle_batch",
        [](DrawList& self,
           nb::ndarray<float, nb::shape<-1, 2>> center,
           nb::ndarray<float, nb::shape<-1>> radius,
           nb::ndarray<uint32_t, nb::shape<-1>> color,
           int num_segments,
           float thickness
        ) {
            bool col_per_object = color.shape(0) != 1;
            bool radius_per_object = radius.shape(0) != 1;
            for (size_t i = 0; i < center.shape(0); i++) {
                self.list->AddCircle(
                    ImVec2(center(i, 0), center(i, 1)),
                    radius_per_object ? radius(i) : radius(0),
                    col_per_object ? color(i) : color(0),
                    num_segments,
                    thickness
                );
            }
        },
        nb::arg("center"),
        nb::arg("radius"),
        nb::arg("color"),
        nb::arg("num_segments") = 0,
        nb::arg("thickness") = 1.0f
    );

    drawlist_class.def("add_circle_filled_batch",
        [](DrawList& self,
           nb::ndarray<float, nb::shape<-1, 2>> center,
           nb::ndarray<float, nb::shape<-1>> radius,
           nb::ndarray<uint32_t, nb::shape<-1>> color,
           int num_segments
        ) {
            bool col_per_object = color.shape(0) != 1;
            bool radius_per_object = radius.shape(0) != 1;
            for (size_t i = 0; i < center.shape(0); i++) {
                self.list->AddCircleFilled(
                    ImVec2(center(i, 0), center(i, 1)),
                    radius_per_object ? radius(i) : radius(0),
                    col_per_object ? color(i) : color(0),
                    num_segments
                );
            }
        },
        nb::arg("center"),
        nb::arg("radius"),
        nb::arg("color"),
        nb::arg("num_segments") = 0
    );


    // IO
    mod_imgui.def("get_io", ImGui::GetIO, nb::rv_policy::reference);
    mod_imgui.def("get_style", ImGui::GetStyle, nb::rv_policy::reference);
    mod_imgui.def("push_font",
        [] (std::optional<Font*> font, float font_size_base_unscaled) -> void {
            return ImGui::PushFont(font.has_value() ? (*font)->font : NULL, font_size_base_unscaled);
        }, nb::arg("font").none(), nb::arg("font_size_base_unscaled") = 0.0f);

    // APIs not covered by generated bindings
    mod_imgui.def("set_next_window_size_constraints", [](const ImVec2& min_size, const ImVec2& max_size) { ImGui::SetNextWindowSizeConstraints(min_size, max_size); }, nb::arg("min_size"), nb::arg("max_size"));

    // Texture
    nb::class_<Texture>(mod_imgui, "Texture",
        nb::intrusive_ptr<Texture>([](Texture *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<DescriptorSet>>(), nb::arg("descriptor_set"))
        .def("destroy", &Texture::destroy)
    ;

    // Should this be pyxpg.imgui.implot or pyxpg.implot?
    nb::module_ mod_implot = mod_imgui.def_submodule("implot", "ImPlot bindings for XPG");
    #include "generated_implot.inc"

    mod_implot.def("plot_heatmap",
        [] (nb::str label_id, nb::ndarray<nb::ndim<2>, nb::c_contig, nb::device::cpu> values, double scale_min, double scale_max, nb::str label_fmt, double min_x, double min_y, double max_x, double max_y, ImPlotHeatmapFlags_ flags) {
            size_t rows = values.shape(0);
            size_t cols = values.shape(1);
            nb::dlpack::dtype dtype = values.dtype();
            ImPlotSpec spec;
            spec.Flags = flags;
            switch (dtype.code) {
                case (int)nb::dlpack::dtype_code::Int: {
                    switch (dtype.bits) {
                        case  8: ImPlot::PlotHeatmap(label_id.c_str(), (ImS8*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                        case 16: ImPlot::PlotHeatmap(label_id.c_str(), (ImS16*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                        case 32: ImPlot::PlotHeatmap(label_id.c_str(), (ImS32*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                        case 64: ImPlot::PlotHeatmap(label_id.c_str(), (ImS64*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                    }
                } break;
                case (int)nb::dlpack::dtype_code::UInt: {
                    switch (dtype.bits) {
                        case  8: ImPlot::PlotHeatmap(label_id.c_str(), (ImU8*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                        case 16: ImPlot::PlotHeatmap(label_id.c_str(), (ImU16*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                        case 32: ImPlot::PlotHeatmap(label_id.c_str(), (ImU32*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                        case 64: ImPlot::PlotHeatmap(label_id.c_str(), (ImU64*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                    }
                } break;
                case (int)nb::dlpack::dtype_code::Float: {
                    switch (dtype.bits) {
                        case 32: ImPlot::PlotHeatmap(label_id.c_str(), (float*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                        case 64: ImPlot::PlotHeatmap(label_id.c_str(), (double*)values.data(), rows, cols, scale_min, scale_max, label_fmt.c_str(), ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), spec); break;
                    }
                } break;
                default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
            }
        }, nb::arg("label_id"), nb::arg("values"), nb::arg("scale_min") = 0, nb::arg("scale_max") = 0, nb::arg("label_fmt") = "%.1f", nb::arg("min_x") = 0, nb::arg("min_y") = 0, nb::arg("max_x") = 1, nb::arg("max_y") = 1, nb::arg("flags") = ImPlotHeatmapFlags_None);

    mod_implot.def("plot_image",
        [] (nb::str label_id, const Texture& texture, double min_x, double min_y, double max_x, double max_y, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, ImPlotLineFlags_ flags) {
            ImPlotSpec spec;
            spec.Flags = flags;
            ImPlot::PlotImage(label_id.c_str(), texture.tex_ref, ImPlotPoint(min_x, min_y), ImPlotPoint(max_x, max_y), uv0, uv1, tint_col, spec);
        }, nb::arg("name"), nb::arg("texture"), nb::arg("min_x"), nb::arg("min_y"), nb::arg("max_x"), nb::arg("max_y"), nb::arg("uv0") = ImVec2(0, 0), nb::arg("uv1") = ImVec2(1, 1), nb::arg("tint_col") = ImVec4(1, 1, 1, 1), nb::arg("flags") = ImPlotLineFlags_None);

    mod_implot.def("plot_text",
        [] (nb::str text, float x, float y, const ImVec2& pix_offset, ImPlotTextFlags_ flags) -> void {
            ImPlotSpec spec;
            spec.Flags = flags;
            ImPlot::PlotText(text.c_str(), x, y, pix_offset, spec);
        }, nb::arg("text"), nb::arg("x"), nb::arg("y"), nb::arg("pix_offset") = ImVec2(0, 0), nb::arg("flags") = 0);

    mod_implot.def("plot_dummy",
        [] (nb::str label_id, ImPlotDummyFlags_ flags) -> void {
            ImPlotSpec spec;
            spec.Flags = flags;
            return ImPlot::PlotDummy(label_id.c_str(), spec);
        }, nb::arg("label_id"), nb::arg("flags") = 0);
    

}

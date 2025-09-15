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

    // Texture
    nb::class_<Texture>(mod_imgui, "Texture",
        nb::intrusive_ptr<Texture>([](Texture *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<DescriptorSet>>(), nb::arg("descriptor_set"))
        .def("destroy", &Texture::destroy)
    ;

    // Should this be pyxpg.imgui.implot or pyxpg.implot?
    nb::module_ mod_implot = mod_imgui.def_submodule("implot", "ImPlot bindings for XPG");
    #include "generated_implot.inc"

    #define ONE_ARRAY(func, ...) \
            size_t ndim = values.ndim(); \
            if (ndim != 1) { \
                nb::raise("Invalid array shape for parameter \"values\". Expected 1 dimension, got %zu", ndim); \
            } \
            size_t stride = values.stride(0); \
            size_t count = values.shape(0); \
            nb::dlpack::dtype dtype = values.dtype(); \
            switch (dtype.code) { \
                case (int)nb::dlpack::dtype_code::Int: { \
                    switch(dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (int8_t *)values.data(), count, __VA_ARGS__, 0, stride * 1); break; \
                        case 16: ImPlot::func(name.c_str(), (int16_t*)values.data(), count, __VA_ARGS__, 0, stride * 2); break; \
                        case 32: ImPlot::func(name.c_str(), (int32_t*)values.data(), count, __VA_ARGS__, 0, stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (int64_t*)values.data(), count, __VA_ARGS__, 0, stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::UInt: { \
                    switch(dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (uint8_t *)values.data(), count, __VA_ARGS__, 0, stride * 1); break; \
                        case 16: ImPlot::func(name.c_str(), (uint16_t*)values.data(), count, __VA_ARGS__, 0, stride * 2); break; \
                        case 32: ImPlot::func(name.c_str(), (uint32_t*)values.data(), count, __VA_ARGS__, 0, stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (uint64_t*)values.data(), count, __VA_ARGS__, 0, stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::Float: { \
                    switch(dtype.bits) { \
                        case 32: ImPlot::func(name.c_str(), (float *)values.data(), count, __VA_ARGS__, 0, stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (double*)values.data(), count, __VA_ARGS__, 0, stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); break; \
                    } \
                } break; \
                default: { \
                    nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); \
                } break; \
            }

    #define ONE_ARRAY_NO_OFFSET_STRIDE(func, ...) \
            size_t ndim = values.ndim(); \
            if (ndim != 1) { \
                nb::raise("Invalid array shape for parameter \"values\". Expected 1 dimension, got %zu", ndim); \
            } \
            size_t stride = values.stride(0); \
            size_t count = values.shape(0); \
            nb::dlpack::dtype dtype = values.dtype(); \
            switch (dtype.code) { \
                case (int)nb::dlpack::dtype_code::Int: { \
                    switch(dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (int8_t *)values.data(), count, __VA_ARGS__); break; \
                        case 16: ImPlot::func(name.c_str(), (int16_t*)values.data(), count, __VA_ARGS__); break; \
                        case 32: ImPlot::func(name.c_str(), (int32_t*)values.data(), count, __VA_ARGS__); break; \
                        case 64: ImPlot::func(name.c_str(), (int64_t*)values.data(), count, __VA_ARGS__); break; \
                        default: nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::UInt: { \
                    switch(dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (uint8_t *)values.data(), count, __VA_ARGS__); break; \
                        case 16: ImPlot::func(name.c_str(), (uint16_t*)values.data(), count, __VA_ARGS__); break; \
                        case 32: ImPlot::func(name.c_str(), (uint32_t*)values.data(), count, __VA_ARGS__); break; \
                        case 64: ImPlot::func(name.c_str(), (uint64_t*)values.data(), count, __VA_ARGS__); break; \
                        default: nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::Float: { \
                    switch(dtype.bits) { \
                        case 32: ImPlot::func(name.c_str(), (float *)values.data(), count, __VA_ARGS__); break; \
                        case 64: ImPlot::func(name.c_str(), (double*)values.data(), count, __VA_ARGS__); break; \
                        default: nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); break; \
                    } \
                } break; \
                default: { \
                    nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); \
                } break; \
            }

    #define TWO_ARRAYS(func, ...) \
            size_t x_ndim = xs.ndim(); \
            if (x_ndim != 1) { \
                nb::raise("Invalid array shape for parameter \"xs\". Expected 1 dimension, got %zu", x_ndim); \
            } \
            size_t x_stride = xs.stride(0); \
            size_t x_count = xs.shape(0); \
            nb::dlpack::dtype x_dtype = xs.dtype(); \
            size_t y_ndim = ys.ndim(); \
            if (y_ndim != 1) { \
                nb::raise("Invalid array shape for parameter \"ys\". Expected 1 dimension, got %zu", y_ndim); \
            } \
            size_t y_stride = ys.stride(0); \
            size_t y_count = ys.shape(0); \
            nb::dlpack::dtype y_dtype = ys.dtype(); \
            if (x_dtype != y_dtype) { \
                nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(x_dtype.bits), x_dtype.bits, dtype_code_to_str(y_dtype.code), y_dtype.bits); \
            } \
            if (x_stride != y_stride || x_count != y_count) { \
                nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu) and shape (xs = %zu, ys = %zu)", x_stride, y_stride, x_count, y_count); \
            } \
            switch (x_dtype.code) { \
                case (int)nb::dlpack::dtype_code::Int: { \
                    switch(x_dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (int8_t *)xs.data(), (int8_t *)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 1); break; \
                        case 16: ImPlot::func(name.c_str(), (int16_t*)xs.data(), (int16_t*)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 2); break; \
                        case 32: ImPlot::func(name.c_str(), (int32_t*)xs.data(), (int32_t*)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (int64_t*)xs.data(), (int64_t*)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::UInt: { \
                    switch(x_dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (uint8_t *)xs.data(), (uint8_t *)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 1); break; \
                        case 16: ImPlot::func(name.c_str(), (uint16_t*)xs.data(), (uint16_t*)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 2); break; \
                        case 32: ImPlot::func(name.c_str(), (uint32_t*)xs.data(), (uint32_t*)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (uint64_t*)xs.data(), (uint64_t*)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::Float: { \
                    switch(x_dtype.bits) { \
                        case 32: ImPlot::func(name.c_str(), (float *)xs.data(), (float *)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (double*)xs.data(), (double*)ys.data(), x_count, __VA_ARGS__, 0, x_stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); break; \
                    } \
                } break; \
                default: { \
                    nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); \
                } break; \
            }

    #define THREE_ARRAYS(func, ...) \
            size_t x_ndim = xs.ndim(); \
            if (x_ndim != 1) { \
                nb::raise("Invalid array shape for parameter \"xs\". Expected 1 dimension, got %zu", x_ndim); \
            } \
            size_t x_stride = xs.stride(0); \
            size_t x_count = xs.shape(0); \
            nb::dlpack::dtype x_dtype = xs.dtype(); \
            size_t y_ndim = ys1.ndim(); \
            if (y_ndim != 1) { \
                nb::raise("Invalid array shape for parameter \"ys1\". Expected 1 dimension, got %zu", y_ndim); \
            } \
            size_t y_stride = ys1.stride(0); \
            size_t y_count = ys1.shape(0); \
            nb::dlpack::dtype y_dtype = ys1.dtype(); \
            if (x_dtype != y_dtype) { \
                nb::raise("xs and ys1 array must have the same dtype (xs = %s%u, ys1 = %s%u)", dtype_code_to_str(x_dtype.bits), x_dtype.bits, dtype_code_to_str(y_dtype.code), y_dtype.bits); \
            } \
            if (x_stride != y_stride || y_count != x_count) { \
                nb::raise("xs and ys1 array must have the same stride (xs = %zu, ys1 = %zu) and shape (xs = %zu, ys1 = %zu)", x_stride, y_stride, x_count, y_count); \
            } \
            size_t y2_ndim = ys2.ndim(); \
            if (y2_ndim != 1) { \
                nb::raise("Invalid array shape for parameter \"ys2\". Expected 1 dimension, got %zu", y2_ndim); \
            } \
            size_t y2_stride = ys2.stride(0); \
            size_t y2_count = ys2.shape(0); \
            nb::dlpack::dtype y2_dtype = ys2.dtype(); \
            if (x_dtype != y2_dtype) { \
                nb::raise("xs and ys2 array must have the same dtype (xs = %s%u, ys2 = %s%u)", dtype_code_to_str(x_dtype.bits), x_dtype.bits, dtype_code_to_str(y2_dtype.code), y2_dtype.bits); \
            } \
            if (x_stride != y2_stride || x_count != y2_count) { \
                nb::raise("xs and ys2 array must have the same stride (xs = %zu, ys2 = %zu) and shape (xs = %zu, ys2 = %zu)", x_stride, y2_stride, x_count, y2_count); \
            } \
            switch (x_dtype.code) { \
                case (int)nb::dlpack::dtype_code::Int: { \
                    switch(x_dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (int8_t *)xs.data(), (int8_t *)ys1.data(), (int8_t *)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 1); break; \
                        case 16: ImPlot::func(name.c_str(), (int16_t*)xs.data(), (int16_t*)ys1.data(), (int16_t*)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 2); break; \
                        case 32: ImPlot::func(name.c_str(), (int32_t*)xs.data(), (int32_t*)ys1.data(), (int32_t*)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (int64_t*)xs.data(), (int64_t*)ys1.data(), (int64_t*)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::UInt: { \
                    switch(x_dtype.bits) { \
                        case  8: ImPlot::func(name.c_str(), (uint8_t *)xs.data(), (uint8_t *)ys1.data(), (uint8_t *)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 1); break; \
                        case 16: ImPlot::func(name.c_str(), (uint16_t*)xs.data(), (uint16_t*)ys1.data(), (uint16_t*)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 2); break; \
                        case 32: ImPlot::func(name.c_str(), (uint32_t*)xs.data(), (uint32_t*)ys1.data(), (uint32_t*)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (uint64_t*)xs.data(), (uint64_t*)ys1.data(), (uint64_t*)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); break; \
                    } \
                } break; \
                case (int)nb::dlpack::dtype_code::Float: { \
                    switch(x_dtype.bits) { \
                        case 32: ImPlot::func(name.c_str(), (float *)xs.data(), (float *)ys1.data(), (float *)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 4); break; \
                        case 64: ImPlot::func(name.c_str(), (double*)xs.data(), (double*)ys1.data(), (double*)ys2.data(), x_count, __VA_ARGS__, 0, x_stride * 8); break; \
                        default: nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); break; \
                    } \
                } break; \
                default: { \
                    nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", x_dtype.code, x_dtype.bits); \
                } break; \
            }

    mod_implot.def("plot_line",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           double xscale,
           double xstart,
           ImPlotLineFlags flags
        ) {
            ONE_ARRAY(PlotLine, xscale, xstart, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("xscale") = 1,
        nb::arg("xstart") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_line",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           ImPlotLineFlags flags
        ) {
            TWO_ARRAYS(PlotLine, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_scatter",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           double xscale,
           double xstart,
           ImPlotScatterFlags flags
        ) {
            ONE_ARRAY(PlotScatter, xscale, xstart, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("xscale") = 1,
        nb::arg("xstart") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_scatter",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           ImPlotScatterFlags flags
        ) {
            TWO_ARRAYS(PlotScatter, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_stairs",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           double xscale,
           double xstart,
           ImPlotStairsFlags flags
        ) {
            ONE_ARRAY(PlotStairs, xscale, xstart, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("xscale") = 1,
        nb::arg("xstart") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_stairs",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           ImPlotStairsFlags flags
        ) {
            TWO_ARRAYS(PlotStairs, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_shaded",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           double yref,
           double xscale,
           double xstart,
           ImPlotShadedFlags flags
        ) {
            ONE_ARRAY(PlotShaded, yref, xscale, xstart, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("yref") = 0,
        nb::arg("xscale") = 1,
        nb::arg("xstart") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_shaded",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           ImPlotShadedFlags flags
        ) {
            TWO_ARRAYS(PlotShaded, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_shaded",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys1,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys2,
           ImPlotShadedFlags flags
        ) {
            THREE_ARRAYS(PlotShaded, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys1") = 1,
        nb::arg("ys2") = 1,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_bars",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           double bar_size,
           double shift,
           ImPlotBarsFlags flags
        ) {
            ONE_ARRAY(PlotBars, bar_size, shift, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("bar_size") = 0.67,
        nb::arg("shift") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_bars",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           double bar_size,
           ImPlotBarsFlags flags
        ) {
            TWO_ARRAYS(PlotBars, bar_size, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("bar_size") = 0.67,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_bars",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           double bar_size,
           double shift,
           ImPlotBarsFlags flags
        ) {
            ONE_ARRAY(PlotBars, bar_size, shift, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("bar_size") = 0.67,
        nb::arg("shift") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_bars",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           double bar_size,
           ImPlotBarsFlags flags
        ) {
            TWO_ARRAYS(PlotBars, bar_size, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("bar_size") = 0.67,
        nb::arg("flags") = 0
    );

    // TODO: ys1 and ys2 will be printed in errors instead of ys and err
    mod_implot.def("plot_error_bars",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys1,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys2,
           ImPlotErrorBarsFlags flags
        ) {
            THREE_ARRAYS(PlotErrorBars, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("err"),
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_stems",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           double ref,
           double scale,
           double start,
           ImPlotStemsFlags flags
        ) {
            ONE_ARRAY(PlotStems, ref, scale, start, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("ref") = 0,
        nb::arg("scale") = 1,
        nb::arg("start") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_stems",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           double ref,
           ImPlotStemsFlags flags
        ) {
            TWO_ARRAYS(PlotStems, ref, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("ref") = 0,
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_inf_lines",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           ImPlotInfLinesFlags flags
        ) {
            ONE_ARRAY(PlotInfLines, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_histogram",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> values,
           int bins,
           double bar_scale,
           std::tuple<double, double> range,
           ImPlotHistogramFlags flags
        ) {
            if (values.stride(0) * 8 != values.dtype().bits) {
                nb::raise("Array \"values\" must be C-contiguos");
            }
            ImPlotRange implot_range(std::get<0>(range), std::get<1>(range));
            ONE_ARRAY_NO_OFFSET_STRIDE(PlotHistogram, bins, bar_scale, implot_range, flags)
        },
        nb::arg("label_id"),
        nb::arg("values"),
        nb::arg("bins") = ImPlotBin_Sturges,
        nb::arg("bar_scale") = 1.0,
        nb::arg("range") = std::make_tuple(0, 0),
        nb::arg("flags") = 0
    );

    mod_implot.def("plot_digital",
        [](nb::str name,
           nb::ndarray<nb::any_contig, nb::device::cpu> xs,
           nb::ndarray<nb::any_contig, nb::device::cpu> ys,
           ImPlotStemsFlags flags
        ) {
            TWO_ARRAYS(PlotDigital, flags)
        },
        nb::arg("label_id"),
        nb::arg("xs"),
        nb::arg("ys"),
        nb::arg("flags") = 0
    );

}

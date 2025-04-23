#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>
#include <nanobind/operators.h>


#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>                      // GLM math
#include <glm/gtc/matrix_transform.hpp>     // GLM matrix ops
#include <glm/gtx/norm.hpp>                 // length2, distance and distance2

namespace nb = nanobind;
using namespace glm;

glm::vec2 test(glm::vec2 v) {
    return v * 3.0f;
}

glm::ivec2 testi(glm::ivec2 v) {
    return v * 3;
}


void math_create_bindings(nb::module_& mod_math) {
    #include "generated_math.inc"

    // List of things to generate
    // vec: (2, 3, 4 | i, f to start | d, h maybe also useful)
    //  Constructors:
    //   - scalar
    //   - all vector types of same or bigger size
    //   - same vector types of smaller size padded with scalars (or 2 vec2s)
    //   - tuples and lists (also define implicit conversion)
    //  RW props:
    //   - x, y, z, w (also alias for rgb?)
    //  Slots:
    //   - __iter__: conversion to list
    //   - __repr__: conversion to string
    //  Operators:
    //   - Arithmetic + - * / 
    //   - Unary -
    //   - % (int only)
    //   - Bitwise & | ^ ~ (int only)
    //  Numpy array compat
    // mat:
    //   - TODO
    // quaternion:
    //   - TODO
    // funcs (prefer not-method style):
    //  - length, normalize, dot, cross, square, pow, sqrt (float only)
    //  - transformations
    //      - translation
    //      - lookat / perspective
#if 0
    // nb::class_<glm::vec2>(mod_math, "vec2")
    //     .def(nb::init<>())
    //     .def(nb::init<float, float>())
    //     .def_rw("x", &glm::vec2::x)
    //     .def_rw("y", &glm::vec2::y)
    // ;

    nb::class_<glm::vec2>(mod_math, "vec2")
        .def(nb::init<>())
        .def(nb::init<float, float>())
        .def(nb::init<glm::ivec2>())
        .def("__init__", [](glm::vec2* v, nb::tuple t) {
            if (t.size() != 2) {
                nb::raise_type_error("Cannot convert tuple of length %zu to vec2", t.size());
            }
            new (v) glm::vec2(nb::cast<float>(t[0]), nb::cast<float>(t[1]));
        })
        .def("__init__", [](glm::vec2* v, nb::list l) {
            if (l.size() != 2) {
                nb::raise_type_error("Cannot convert list of length %zu to ivec2", l.size());
            }
            new (v) glm::vec2(nb::cast<float>(l[0]), nb::cast<float>(l[1]));
        })
        .def_rw("x", &glm::vec2::x)
        .def_rw("y", &glm::vec2::y)
        .def("__repr__", [](const glm::vec2& v) { 
            char buf[128];
            snprintf(buf, sizeof(buf), "vec2(%g, %g)", v.x, v.y);
            return nb::str(buf);
        })
        .def("__iter__", [](const glm::vec2 &v) {
            return nb::make_iterator(nb::type<glm::vec2>(), "vec2_iterator", &v.x, &v.x + 2);
        })
        .def(nb::self +  nb::self)
        .def(nb::self += nb::self)
        .def(nb::self +  float())
        .def(float()  +  nb::self)
        .def(nb::self -  nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self -  float())
        .def(float()  -  nb::self)
        .def(nb::self *  nb::self)
        .def(nb::self *  float())
        .def(float()  *  nb::self)
        .def(nb::self *= float())
        .def(nb::self /  nb::self)
        .def(nb::self /  float())
        .def(float()  /  nb::self)
        .def(nb::self /= float())
        // .def(nb::self % nb::self)
        // .def(nb::self % float())
        // .def(float() % nb::self)
        // .def(nb::self %= float())
        .def(-nb::self)
    ;
    nb::implicitly_convertible<nb::tuple, glm::vec2>();
    nb::implicitly_convertible<nb::list, glm::vec2>();

    nb::class_<glm::ivec2>(mod_math, "ivec2")
        .def(nb::init<>())
        .def(nb::init<int, int>())
        .def(nb::init<glm::ivec3>())
        .def("__init__", [](glm::ivec2* v, nb::tuple t) {
            if (t.size() != 2) {
                nb::raise_type_error("Cannot convert tuple of length %zu to ivec2", t.size());
            }
            new (v) glm::ivec2(nb::cast<int>(t[0]), nb::cast<int>(t[1]));
        })
        .def("__init__", [](glm::ivec2* v, nb::list l) {
            if (l.size() != 2) {
                nb::raise_type_error("Cannot convert list of length %zu to ivec2", l.size());
            }
            new (v) glm::ivec2(nb::cast<int>(l[0]), nb::cast<int>(l[1]));
        })
        .def_rw("x", &glm::ivec2::x)
        .def_rw("y", &glm::ivec2::y)
        .def("__repr__", [](const glm::ivec2& v) { 
            char buf[128];
            snprintf(buf, sizeof(buf), "ivec2(%d, %d)", v.x, v.y);
            return nb::str(buf);
        })
        .def("__iter__", [](const glm::ivec2 &v) {
            return nb::make_iterator(nb::type<glm::ivec2>(), "ivec2_iterator", &v.x, &v.x + 2);
        })
        .def(nb::self + nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + int())
        .def(int() + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self - int())
        .def(int() - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self * int())
        .def(int() * nb::self)
        .def(nb::self *= int())
        .def(nb::self / nb::self)
        .def(nb::self / int())
        .def(int() / nb::self)
        .def(nb::self /= int())
        .def(nb::self % nb::self)
        .def(nb::self % int())
        .def(int() % nb::self)
        .def(nb::self %= int())
        .def(-nb::self)
    ;
    nb::implicitly_convertible<nb::tuple, glm::ivec2>();
    nb::implicitly_convertible<nb::list, glm::ivec2>();

    nb::class_<glm::ivec3>(mod_math, "ivec3")
        .def(nb::init<>())
        .def(nb::init<int>())
        .def(nb::init<int, int, int>())
        .def(nb::init<glm::ivec2, int>())
        .def(nb::init<int, glm::ivec2>())
        .def("__init__", [](glm::ivec3* v, nb::tuple t) {
            if (t.size() != 3) {
                nb::raise_type_error("Cannot convert tuple of length %zu to ivec3", t.size());
            }
            new (v) glm::ivec3(nb::cast<int>(t[0]), nb::cast<int>(t[1]), nb::cast<int>(t[2]));
        })
        .def("__init__", [](glm::ivec3* v, nb::list l) {
            if (l.size() != 3) {
                nb::raise_type_error("Cannot convert list of length %zu to ivec3", l.size());
            }
            new (v) glm::ivec3(nb::cast<int>(l[0]), nb::cast<int>(l[1]), nb::cast<int>(l[2]));
        })
        .def_rw("x", &glm::ivec3::x)
        .def_rw("y", &glm::ivec3::y)
        .def_rw("y", &glm::ivec3::z)
        .def("__repr__", [](const glm::ivec3& v) { 
            char buf[128];
            snprintf(buf, sizeof(buf), "ivec3(%d, %d, %d)", v.x, v.y, v.z);
            return nb::str(buf);
        })
        .def("__iter__", [](const glm::ivec3 &v) {
            return nb::make_iterator(nb::type<glm::ivec3>(), "ivec3_iterator", &v.x, &v.x + 3);
        })
        .def(nb::self + nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + int())
        .def(int() + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self - int())
        .def(int() - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self * int())
        .def(int() * nb::self)
        .def(nb::self *= int())
        .def(nb::self / nb::self)
        .def(nb::self / int())
        .def(int() / nb::self)
        .def(nb::self /= int())
        .def(nb::self % nb::self)
        .def(nb::self % int())
        .def(int() % nb::self)
        .def(nb::self %= int())
        .def(-nb::self)
    ;
    nb::implicitly_convertible<nb::tuple, glm::ivec3>();
    nb::implicitly_convertible<nb::list, glm::ivec3>();

    mod_math.def("test", test);
    mod_math.def("testi", testi);
#endif
}
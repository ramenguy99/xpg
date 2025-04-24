nb::class_<vec2>(mod_math, "vec2")
    .def(nb::init<>())
    .def(nb::init<float, float>(), nb::arg("x"), nb::arg("y"))
    .def(nb::init<vec2>(), nb::arg("v"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec2>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec2>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec2>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](vec2* v, nb::tuple t) {
        if (t.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to vec2", t.size());
        }
        new (v) vec2(nb::cast<float>(t[0]), nb::cast<float>(t[1]));
    }, nb::arg("t"))
    .def("__init__", [](vec2* v, nb::list l) {
        if (l.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to vec2", l.size());
        }
        new (v) vec2(nb::cast<float>(l[0]), nb::cast<float>(l[1]));
    }, nb::arg("l"))
    .def_rw("x", &vec2::x, nb::arg("val"))
    .def_rw("y", &vec2::y, nb::arg("val"))
    .def("__repr__", [](const vec2& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "vec2(%g, %g)", v.x, v.y);
        return nb::str(buf);
    })
    .def("__iter__", [](const vec2 &v) {
        return nb::make_iterator(nb::type<vec2>(), "vec2_iterator", &v.x, &v.x + 2);
    })
    .def("__array__", [] (const vec2& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const float, nb::numpy, nb::shape<2>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  float (), nb::arg("other"))
    .def(nb::self += float (), nb::arg("other"))
    .def(float () +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  float (), nb::arg("other"))
    .def(nb::self -= float (), nb::arg("other"))
    .def(float () -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  float (), nb::arg("other"))
    .def(nb::self *= float (), nb::arg("other"))
    .def(float () *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  float (), nb::arg("other"))
    .def(nb::self /= float (), nb::arg("other"))
    .def(float () /  nb::self, nb::arg("other"))
    .def(-nb::self)
;
nb::implicitly_convertible<nb::tuple, vec2>();
nb::implicitly_convertible<nb::list, vec2>();
mod_math.def("length", [](const vec2& v) { return length(v); }, nb::arg("v"));
mod_math.def("length2", [](const vec2& v) { return length2(v); }, nb::arg("v"));
mod_math.def("normalize", [](const vec2& v) { return normalize(v); }, nb::arg("v"));
mod_math.def("dot", [](const vec2& a, const vec2& b) { return dot(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance", [](const vec2& a, const vec2& b) { return distance(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance2", [](const vec2& a, const vec2& b) { return distance2(a, b); }, nb::arg("a"), nb::arg("b"));

nb::class_<vec3>(mod_math, "vec3")
    .def(nb::init<>())
    .def(nb::init<float, float, float>(), nb::arg("x"), nb::arg("y"), nb::arg("z"))
    .def(nb::init<float, vec2>(), nb::arg("x"), nb::arg("yz"))
    .def(nb::init<vec2, float>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](vec3* v, nb::tuple t) {
        if (t.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to vec3", t.size());
        }
        new (v) vec3(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]));
    }, nb::arg("t"))
    .def("__init__", [](vec3* v, nb::list l) {
        if (l.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to vec3", l.size());
        }
        new (v) vec3(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]));
    }, nb::arg("l"))
    .def_rw("x", &vec3::x, nb::arg("val"))
    .def_rw("y", &vec3::y, nb::arg("val"))
    .def_rw("z", &vec3::z, nb::arg("val"))
    .def("__repr__", [](const vec3& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "vec3(%g, %g, %g)", v.x, v.y, v.z);
        return nb::str(buf);
    })
    .def("__iter__", [](const vec3 &v) {
        return nb::make_iterator(nb::type<vec3>(), "vec3_iterator", &v.x, &v.x + 3);
    })
    .def("__array__", [] (const vec3& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const float, nb::numpy, nb::shape<3>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  float (), nb::arg("other"))
    .def(nb::self += float (), nb::arg("other"))
    .def(float () +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  float (), nb::arg("other"))
    .def(nb::self -= float (), nb::arg("other"))
    .def(float () -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  float (), nb::arg("other"))
    .def(nb::self *= float (), nb::arg("other"))
    .def(float () *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  float (), nb::arg("other"))
    .def(nb::self /= float (), nb::arg("other"))
    .def(float () /  nb::self, nb::arg("other"))
    .def(-nb::self)
;
nb::implicitly_convertible<nb::tuple, vec3>();
nb::implicitly_convertible<nb::list, vec3>();
mod_math.def("length", [](const vec3& v) { return length(v); }, nb::arg("v"));
mod_math.def("length2", [](const vec3& v) { return length2(v); }, nb::arg("v"));
mod_math.def("normalize", [](const vec3& v) { return normalize(v); }, nb::arg("v"));
mod_math.def("dot", [](const vec3& a, const vec3& b) { return dot(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance", [](const vec3& a, const vec3& b) { return distance(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance2", [](const vec3& a, const vec3& b) { return distance2(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("cross", [](const vec3& a, const vec3& b) { return cross(a, b); }, nb::arg("a"), nb::arg("b"));

nb::class_<vec4>(mod_math, "vec4")
    .def(nb::init<>())
    .def(nb::init<float, float, float, float>(), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<float, vec3>(), nb::arg("x"), nb::arg("yzw"))
    .def(nb::init<vec3, float>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<float, float, vec2>(), nb::arg("x"), nb::arg("y"), nb::arg("zw"))
    .def(nb::init<float, vec2, float>(), nb::arg("x"), nb::arg("yz"), nb::arg("w"))
    .def(nb::init<vec2, float, float>(), nb::arg("xy"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](vec4* v, nb::tuple t) {
        if (t.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to vec4", t.size());
        }
        new (v) vec4(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]));
    }, nb::arg("t"))
    .def("__init__", [](vec4* v, nb::list l) {
        if (l.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to vec4", l.size());
        }
        new (v) vec4(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]));
    }, nb::arg("l"))
    .def_rw("x", &vec4::x, nb::arg("val"))
    .def_rw("y", &vec4::y, nb::arg("val"))
    .def_rw("z", &vec4::z, nb::arg("val"))
    .def_rw("w", &vec4::w, nb::arg("val"))
    .def("__repr__", [](const vec4& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "vec4(%g, %g, %g, %g)", v.x, v.y, v.z, v.w);
        return nb::str(buf);
    })
    .def("__iter__", [](const vec4 &v) {
        return nb::make_iterator(nb::type<vec4>(), "vec4_iterator", &v.x, &v.x + 4);
    })
    .def("__array__", [] (const vec4& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const float, nb::numpy, nb::shape<4>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  float (), nb::arg("other"))
    .def(nb::self += float (), nb::arg("other"))
    .def(float () +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  float (), nb::arg("other"))
    .def(nb::self -= float (), nb::arg("other"))
    .def(float () -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  float (), nb::arg("other"))
    .def(nb::self *= float (), nb::arg("other"))
    .def(float () *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  float (), nb::arg("other"))
    .def(nb::self /= float (), nb::arg("other"))
    .def(float () /  nb::self, nb::arg("other"))
    .def(-nb::self)
;
nb::implicitly_convertible<nb::tuple, vec4>();
nb::implicitly_convertible<nb::list, vec4>();
mod_math.def("length", [](const vec4& v) { return length(v); }, nb::arg("v"));
mod_math.def("length2", [](const vec4& v) { return length2(v); }, nb::arg("v"));
mod_math.def("normalize", [](const vec4& v) { return normalize(v); }, nb::arg("v"));
mod_math.def("dot", [](const vec4& a, const vec4& b) { return dot(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance", [](const vec4& a, const vec4& b) { return distance(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance2", [](const vec4& a, const vec4& b) { return distance2(a, b); }, nb::arg("a"), nb::arg("b"));

nb::class_<dvec2>(mod_math, "dvec2")
    .def(nb::init<>())
    .def(nb::init<double, double>(), nb::arg("x"), nb::arg("y"))
    .def(nb::init<vec2>(), nb::arg("v"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec2>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec2>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec2>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](dvec2* v, nb::tuple t) {
        if (t.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to dvec2", t.size());
        }
        new (v) dvec2(nb::cast<float>(t[0]), nb::cast<float>(t[1]));
    }, nb::arg("t"))
    .def("__init__", [](dvec2* v, nb::list l) {
        if (l.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to dvec2", l.size());
        }
        new (v) dvec2(nb::cast<float>(l[0]), nb::cast<float>(l[1]));
    }, nb::arg("l"))
    .def_rw("x", &dvec2::x, nb::arg("val"))
    .def_rw("y", &dvec2::y, nb::arg("val"))
    .def("__repr__", [](const dvec2& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dvec2(%g, %g)", v.x, v.y);
        return nb::str(buf);
    })
    .def("__iter__", [](const dvec2 &v) {
        return nb::make_iterator(nb::type<dvec2>(), "dvec2_iterator", &v.x, &v.x + 2);
    })
    .def("__array__", [] (const dvec2& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const double, nb::numpy, nb::shape<2>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  double(), nb::arg("other"))
    .def(nb::self += double(), nb::arg("other"))
    .def(double() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  double(), nb::arg("other"))
    .def(nb::self -= double(), nb::arg("other"))
    .def(double() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  double(), nb::arg("other"))
    .def(nb::self *= double(), nb::arg("other"))
    .def(double() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  double(), nb::arg("other"))
    .def(nb::self /= double(), nb::arg("other"))
    .def(double() /  nb::self, nb::arg("other"))
    .def(-nb::self)
;
nb::implicitly_convertible<nb::tuple, dvec2>();
nb::implicitly_convertible<nb::list, dvec2>();
mod_math.def("length", [](const dvec2& v) { return length(v); }, nb::arg("v"));
mod_math.def("length2", [](const dvec2& v) { return length2(v); }, nb::arg("v"));
mod_math.def("normalize", [](const dvec2& v) { return normalize(v); }, nb::arg("v"));
mod_math.def("dot", [](const dvec2& a, const dvec2& b) { return dot(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance", [](const dvec2& a, const dvec2& b) { return distance(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance2", [](const dvec2& a, const dvec2& b) { return distance2(a, b); }, nb::arg("a"), nb::arg("b"));

nb::class_<dvec3>(mod_math, "dvec3")
    .def(nb::init<>())
    .def(nb::init<double, double, double>(), nb::arg("x"), nb::arg("y"), nb::arg("z"))
    .def(nb::init<double, dvec2>(), nb::arg("x"), nb::arg("yz"))
    .def(nb::init<dvec2, double>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](dvec3* v, nb::tuple t) {
        if (t.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to dvec3", t.size());
        }
        new (v) dvec3(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]));
    }, nb::arg("t"))
    .def("__init__", [](dvec3* v, nb::list l) {
        if (l.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to dvec3", l.size());
        }
        new (v) dvec3(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]));
    }, nb::arg("l"))
    .def_rw("x", &dvec3::x, nb::arg("val"))
    .def_rw("y", &dvec3::y, nb::arg("val"))
    .def_rw("z", &dvec3::z, nb::arg("val"))
    .def("__repr__", [](const dvec3& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dvec3(%g, %g, %g)", v.x, v.y, v.z);
        return nb::str(buf);
    })
    .def("__iter__", [](const dvec3 &v) {
        return nb::make_iterator(nb::type<dvec3>(), "dvec3_iterator", &v.x, &v.x + 3);
    })
    .def("__array__", [] (const dvec3& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const double, nb::numpy, nb::shape<3>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  double(), nb::arg("other"))
    .def(nb::self += double(), nb::arg("other"))
    .def(double() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  double(), nb::arg("other"))
    .def(nb::self -= double(), nb::arg("other"))
    .def(double() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  double(), nb::arg("other"))
    .def(nb::self *= double(), nb::arg("other"))
    .def(double() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  double(), nb::arg("other"))
    .def(nb::self /= double(), nb::arg("other"))
    .def(double() /  nb::self, nb::arg("other"))
    .def(-nb::self)
;
nb::implicitly_convertible<nb::tuple, dvec3>();
nb::implicitly_convertible<nb::list, dvec3>();
mod_math.def("length", [](const dvec3& v) { return length(v); }, nb::arg("v"));
mod_math.def("length2", [](const dvec3& v) { return length2(v); }, nb::arg("v"));
mod_math.def("normalize", [](const dvec3& v) { return normalize(v); }, nb::arg("v"));
mod_math.def("dot", [](const dvec3& a, const dvec3& b) { return dot(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance", [](const dvec3& a, const dvec3& b) { return distance(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance2", [](const dvec3& a, const dvec3& b) { return distance2(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("cross", [](const dvec3& a, const dvec3& b) { return cross(a, b); }, nb::arg("a"), nb::arg("b"));

nb::class_<dvec4>(mod_math, "dvec4")
    .def(nb::init<>())
    .def(nb::init<double, double, double, double>(), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<double, dvec3>(), nb::arg("x"), nb::arg("yzw"))
    .def(nb::init<dvec3, double>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<double, double, dvec2>(), nb::arg("x"), nb::arg("y"), nb::arg("zw"))
    .def(nb::init<double, dvec2, double>(), nb::arg("x"), nb::arg("yz"), nb::arg("w"))
    .def(nb::init<dvec2, double, double>(), nb::arg("xy"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](dvec4* v, nb::tuple t) {
        if (t.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to dvec4", t.size());
        }
        new (v) dvec4(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]));
    }, nb::arg("t"))
    .def("__init__", [](dvec4* v, nb::list l) {
        if (l.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to dvec4", l.size());
        }
        new (v) dvec4(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]));
    }, nb::arg("l"))
    .def_rw("x", &dvec4::x, nb::arg("val"))
    .def_rw("y", &dvec4::y, nb::arg("val"))
    .def_rw("z", &dvec4::z, nb::arg("val"))
    .def_rw("w", &dvec4::w, nb::arg("val"))
    .def("__repr__", [](const dvec4& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dvec4(%g, %g, %g, %g)", v.x, v.y, v.z, v.w);
        return nb::str(buf);
    })
    .def("__iter__", [](const dvec4 &v) {
        return nb::make_iterator(nb::type<dvec4>(), "dvec4_iterator", &v.x, &v.x + 4);
    })
    .def("__array__", [] (const dvec4& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const double, nb::numpy, nb::shape<4>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  double(), nb::arg("other"))
    .def(nb::self += double(), nb::arg("other"))
    .def(double() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  double(), nb::arg("other"))
    .def(nb::self -= double(), nb::arg("other"))
    .def(double() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  double(), nb::arg("other"))
    .def(nb::self *= double(), nb::arg("other"))
    .def(double() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  double(), nb::arg("other"))
    .def(nb::self /= double(), nb::arg("other"))
    .def(double() /  nb::self, nb::arg("other"))
    .def(-nb::self)
;
nb::implicitly_convertible<nb::tuple, dvec4>();
nb::implicitly_convertible<nb::list, dvec4>();
mod_math.def("length", [](const dvec4& v) { return length(v); }, nb::arg("v"));
mod_math.def("length2", [](const dvec4& v) { return length2(v); }, nb::arg("v"));
mod_math.def("normalize", [](const dvec4& v) { return normalize(v); }, nb::arg("v"));
mod_math.def("dot", [](const dvec4& a, const dvec4& b) { return dot(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance", [](const dvec4& a, const dvec4& b) { return distance(a, b); }, nb::arg("a"), nb::arg("b"));
mod_math.def("distance2", [](const dvec4& a, const dvec4& b) { return distance2(a, b); }, nb::arg("a"), nb::arg("b"));

nb::class_<ivec2>(mod_math, "ivec2")
    .def(nb::init<>())
    .def(nb::init<int32_t, int32_t>(), nb::arg("x"), nb::arg("y"))
    .def(nb::init<vec2>(), nb::arg("v"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec2>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec2>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec2>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](ivec2* v, nb::tuple t) {
        if (t.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to ivec2", t.size());
        }
        new (v) ivec2(nb::cast<float>(t[0]), nb::cast<float>(t[1]));
    }, nb::arg("t"))
    .def("__init__", [](ivec2* v, nb::list l) {
        if (l.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to ivec2", l.size());
        }
        new (v) ivec2(nb::cast<float>(l[0]), nb::cast<float>(l[1]));
    }, nb::arg("l"))
    .def_rw("x", &ivec2::x, nb::arg("val"))
    .def_rw("y", &ivec2::y, nb::arg("val"))
    .def("__repr__", [](const ivec2& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "ivec2(%d, %d)", v.x, v.y);
        return nb::str(buf);
    })
    .def("__iter__", [](const ivec2 &v) {
        return nb::make_iterator(nb::type<ivec2>(), "ivec2_iterator", &v.x, &v.x + 2);
    })
    .def("__array__", [] (const ivec2& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const int32_t, nb::numpy, nb::shape<2>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  int32_t(), nb::arg("other"))
    .def(nb::self += int32_t(), nb::arg("other"))
    .def(int32_t() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  int32_t(), nb::arg("other"))
    .def(nb::self -= int32_t(), nb::arg("other"))
    .def(int32_t() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  int32_t(), nb::arg("other"))
    .def(nb::self *= int32_t(), nb::arg("other"))
    .def(int32_t() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  int32_t(), nb::arg("other"))
    .def(nb::self /= int32_t(), nb::arg("other"))
    .def(int32_t() /  nb::self, nb::arg("other"))
    .def(nb::self %  nb::self, nb::arg("other"))
    .def(nb::self %= nb::self, nb::arg("other"))
    .def(nb::self %  int32_t(), nb::arg("other"))
    .def(nb::self %= int32_t(), nb::arg("other"))
    .def(int32_t() %  nb::self, nb::arg("other"))
    .def(nb::self &  nb::self, nb::arg("other"))
    .def(nb::self &= nb::self, nb::arg("other"))
    .def(nb::self &  int32_t(), nb::arg("other"))
    .def(nb::self &= int32_t(), nb::arg("other"))
    .def(int32_t() &  nb::self, nb::arg("other"))
    .def(nb::self |  nb::self, nb::arg("other"))
    .def(nb::self |= nb::self, nb::arg("other"))
    .def(nb::self |  int32_t(), nb::arg("other"))
    .def(nb::self |= int32_t(), nb::arg("other"))
    .def(int32_t() |  nb::self, nb::arg("other"))
    .def(nb::self ^  nb::self, nb::arg("other"))
    .def(nb::self ^= nb::self, nb::arg("other"))
    .def(nb::self ^  int32_t(), nb::arg("other"))
    .def(nb::self ^= int32_t(), nb::arg("other"))
    .def(int32_t() ^  nb::self, nb::arg("other"))
    .def(-nb::self)
    .def(~nb::self)
;
nb::implicitly_convertible<nb::tuple, ivec2>();
nb::implicitly_convertible<nb::list, ivec2>();

nb::class_<ivec3>(mod_math, "ivec3")
    .def(nb::init<>())
    .def(nb::init<int32_t, int32_t, int32_t>(), nb::arg("x"), nb::arg("y"), nb::arg("z"))
    .def(nb::init<int32_t, ivec2>(), nb::arg("x"), nb::arg("yz"))
    .def(nb::init<ivec2, int32_t>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](ivec3* v, nb::tuple t) {
        if (t.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to ivec3", t.size());
        }
        new (v) ivec3(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]));
    }, nb::arg("t"))
    .def("__init__", [](ivec3* v, nb::list l) {
        if (l.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to ivec3", l.size());
        }
        new (v) ivec3(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]));
    }, nb::arg("l"))
    .def_rw("x", &ivec3::x, nb::arg("val"))
    .def_rw("y", &ivec3::y, nb::arg("val"))
    .def_rw("z", &ivec3::z, nb::arg("val"))
    .def("__repr__", [](const ivec3& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "ivec3(%d, %d, %d)", v.x, v.y, v.z);
        return nb::str(buf);
    })
    .def("__iter__", [](const ivec3 &v) {
        return nb::make_iterator(nb::type<ivec3>(), "ivec3_iterator", &v.x, &v.x + 3);
    })
    .def("__array__", [] (const ivec3& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const int32_t, nb::numpy, nb::shape<3>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  int32_t(), nb::arg("other"))
    .def(nb::self += int32_t(), nb::arg("other"))
    .def(int32_t() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  int32_t(), nb::arg("other"))
    .def(nb::self -= int32_t(), nb::arg("other"))
    .def(int32_t() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  int32_t(), nb::arg("other"))
    .def(nb::self *= int32_t(), nb::arg("other"))
    .def(int32_t() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  int32_t(), nb::arg("other"))
    .def(nb::self /= int32_t(), nb::arg("other"))
    .def(int32_t() /  nb::self, nb::arg("other"))
    .def(nb::self %  nb::self, nb::arg("other"))
    .def(nb::self %= nb::self, nb::arg("other"))
    .def(nb::self %  int32_t(), nb::arg("other"))
    .def(nb::self %= int32_t(), nb::arg("other"))
    .def(int32_t() %  nb::self, nb::arg("other"))
    .def(nb::self &  nb::self, nb::arg("other"))
    .def(nb::self &= nb::self, nb::arg("other"))
    .def(nb::self &  int32_t(), nb::arg("other"))
    .def(nb::self &= int32_t(), nb::arg("other"))
    .def(int32_t() &  nb::self, nb::arg("other"))
    .def(nb::self |  nb::self, nb::arg("other"))
    .def(nb::self |= nb::self, nb::arg("other"))
    .def(nb::self |  int32_t(), nb::arg("other"))
    .def(nb::self |= int32_t(), nb::arg("other"))
    .def(int32_t() |  nb::self, nb::arg("other"))
    .def(nb::self ^  nb::self, nb::arg("other"))
    .def(nb::self ^= nb::self, nb::arg("other"))
    .def(nb::self ^  int32_t(), nb::arg("other"))
    .def(nb::self ^= int32_t(), nb::arg("other"))
    .def(int32_t() ^  nb::self, nb::arg("other"))
    .def(-nb::self)
    .def(~nb::self)
;
nb::implicitly_convertible<nb::tuple, ivec3>();
nb::implicitly_convertible<nb::list, ivec3>();

nb::class_<ivec4>(mod_math, "ivec4")
    .def(nb::init<>())
    .def(nb::init<int32_t, int32_t, int32_t, int32_t>(), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<int32_t, ivec3>(), nb::arg("x"), nb::arg("yzw"))
    .def(nb::init<ivec3, int32_t>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<int32_t, int32_t, ivec2>(), nb::arg("x"), nb::arg("y"), nb::arg("zw"))
    .def(nb::init<int32_t, ivec2, int32_t>(), nb::arg("x"), nb::arg("yz"), nb::arg("w"))
    .def(nb::init<ivec2, int32_t, int32_t>(), nb::arg("xy"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](ivec4* v, nb::tuple t) {
        if (t.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to ivec4", t.size());
        }
        new (v) ivec4(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]));
    }, nb::arg("t"))
    .def("__init__", [](ivec4* v, nb::list l) {
        if (l.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to ivec4", l.size());
        }
        new (v) ivec4(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]));
    }, nb::arg("l"))
    .def_rw("x", &ivec4::x, nb::arg("val"))
    .def_rw("y", &ivec4::y, nb::arg("val"))
    .def_rw("z", &ivec4::z, nb::arg("val"))
    .def_rw("w", &ivec4::w, nb::arg("val"))
    .def("__repr__", [](const ivec4& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "ivec4(%d, %d, %d, %d)", v.x, v.y, v.z, v.w);
        return nb::str(buf);
    })
    .def("__iter__", [](const ivec4 &v) {
        return nb::make_iterator(nb::type<ivec4>(), "ivec4_iterator", &v.x, &v.x + 4);
    })
    .def("__array__", [] (const ivec4& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const int32_t, nb::numpy, nb::shape<4>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  int32_t(), nb::arg("other"))
    .def(nb::self += int32_t(), nb::arg("other"))
    .def(int32_t() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  int32_t(), nb::arg("other"))
    .def(nb::self -= int32_t(), nb::arg("other"))
    .def(int32_t() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  int32_t(), nb::arg("other"))
    .def(nb::self *= int32_t(), nb::arg("other"))
    .def(int32_t() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  int32_t(), nb::arg("other"))
    .def(nb::self /= int32_t(), nb::arg("other"))
    .def(int32_t() /  nb::self, nb::arg("other"))
    .def(nb::self %  nb::self, nb::arg("other"))
    .def(nb::self %= nb::self, nb::arg("other"))
    .def(nb::self %  int32_t(), nb::arg("other"))
    .def(nb::self %= int32_t(), nb::arg("other"))
    .def(int32_t() %  nb::self, nb::arg("other"))
    .def(nb::self &  nb::self, nb::arg("other"))
    .def(nb::self &= nb::self, nb::arg("other"))
    .def(nb::self &  int32_t(), nb::arg("other"))
    .def(nb::self &= int32_t(), nb::arg("other"))
    .def(int32_t() &  nb::self, nb::arg("other"))
    .def(nb::self |  nb::self, nb::arg("other"))
    .def(nb::self |= nb::self, nb::arg("other"))
    .def(nb::self |  int32_t(), nb::arg("other"))
    .def(nb::self |= int32_t(), nb::arg("other"))
    .def(int32_t() |  nb::self, nb::arg("other"))
    .def(nb::self ^  nb::self, nb::arg("other"))
    .def(nb::self ^= nb::self, nb::arg("other"))
    .def(nb::self ^  int32_t(), nb::arg("other"))
    .def(nb::self ^= int32_t(), nb::arg("other"))
    .def(int32_t() ^  nb::self, nb::arg("other"))
    .def(-nb::self)
    .def(~nb::self)
;
nb::implicitly_convertible<nb::tuple, ivec4>();
nb::implicitly_convertible<nb::list, ivec4>();

nb::class_<uvec2>(mod_math, "uvec2")
    .def(nb::init<>())
    .def(nb::init<uint32_t, uint32_t>(), nb::arg("x"), nb::arg("y"))
    .def(nb::init<vec2>(), nb::arg("v"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec2>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec2>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec2>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](uvec2* v, nb::tuple t) {
        if (t.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to uvec2", t.size());
        }
        new (v) uvec2(nb::cast<float>(t[0]), nb::cast<float>(t[1]));
    }, nb::arg("t"))
    .def("__init__", [](uvec2* v, nb::list l) {
        if (l.size() != 2) {
            nb::raise_type_error("Cannot convert tuple of length %zu to uvec2", l.size());
        }
        new (v) uvec2(nb::cast<float>(l[0]), nb::cast<float>(l[1]));
    }, nb::arg("l"))
    .def_rw("x", &uvec2::x, nb::arg("val"))
    .def_rw("y", &uvec2::y, nb::arg("val"))
    .def("__repr__", [](const uvec2& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "uvec2(%d, %d)", v.x, v.y);
        return nb::str(buf);
    })
    .def("__iter__", [](const uvec2 &v) {
        return nb::make_iterator(nb::type<uvec2>(), "uvec2_iterator", &v.x, &v.x + 2);
    })
    .def("__array__", [] (const uvec2& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const uint32_t, nb::numpy, nb::shape<2>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  uint32_t(), nb::arg("other"))
    .def(nb::self += uint32_t(), nb::arg("other"))
    .def(uint32_t() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  uint32_t(), nb::arg("other"))
    .def(nb::self -= uint32_t(), nb::arg("other"))
    .def(uint32_t() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  uint32_t(), nb::arg("other"))
    .def(nb::self *= uint32_t(), nb::arg("other"))
    .def(uint32_t() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  uint32_t(), nb::arg("other"))
    .def(nb::self /= uint32_t(), nb::arg("other"))
    .def(uint32_t() /  nb::self, nb::arg("other"))
    .def(nb::self %  nb::self, nb::arg("other"))
    .def(nb::self %= nb::self, nb::arg("other"))
    .def(nb::self %  uint32_t(), nb::arg("other"))
    .def(nb::self %= uint32_t(), nb::arg("other"))
    .def(uint32_t() %  nb::self, nb::arg("other"))
    .def(nb::self &  nb::self, nb::arg("other"))
    .def(nb::self &= nb::self, nb::arg("other"))
    .def(nb::self &  uint32_t(), nb::arg("other"))
    .def(nb::self &= uint32_t(), nb::arg("other"))
    .def(uint32_t() &  nb::self, nb::arg("other"))
    .def(nb::self |  nb::self, nb::arg("other"))
    .def(nb::self |= nb::self, nb::arg("other"))
    .def(nb::self |  uint32_t(), nb::arg("other"))
    .def(nb::self |= uint32_t(), nb::arg("other"))
    .def(uint32_t() |  nb::self, nb::arg("other"))
    .def(nb::self ^  nb::self, nb::arg("other"))
    .def(nb::self ^= nb::self, nb::arg("other"))
    .def(nb::self ^  uint32_t(), nb::arg("other"))
    .def(nb::self ^= uint32_t(), nb::arg("other"))
    .def(uint32_t() ^  nb::self, nb::arg("other"))
    .def(-nb::self)
    .def(~nb::self)
;
nb::implicitly_convertible<nb::tuple, uvec2>();
nb::implicitly_convertible<nb::list, uvec2>();

nb::class_<uvec3>(mod_math, "uvec3")
    .def(nb::init<>())
    .def(nb::init<uint32_t, uint32_t, uint32_t>(), nb::arg("x"), nb::arg("y"), nb::arg("z"))
    .def(nb::init<uint32_t, uvec2>(), nb::arg("x"), nb::arg("yz"))
    .def(nb::init<uvec2, uint32_t>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<vec3>(), nb::arg("v"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec3>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec3>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec3>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](uvec3* v, nb::tuple t) {
        if (t.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to uvec3", t.size());
        }
        new (v) uvec3(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]));
    }, nb::arg("t"))
    .def("__init__", [](uvec3* v, nb::list l) {
        if (l.size() != 3) {
            nb::raise_type_error("Cannot convert tuple of length %zu to uvec3", l.size());
        }
        new (v) uvec3(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]));
    }, nb::arg("l"))
    .def_rw("x", &uvec3::x, nb::arg("val"))
    .def_rw("y", &uvec3::y, nb::arg("val"))
    .def_rw("z", &uvec3::z, nb::arg("val"))
    .def("__repr__", [](const uvec3& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "uvec3(%d, %d, %d)", v.x, v.y, v.z);
        return nb::str(buf);
    })
    .def("__iter__", [](const uvec3 &v) {
        return nb::make_iterator(nb::type<uvec3>(), "uvec3_iterator", &v.x, &v.x + 3);
    })
    .def("__array__", [] (const uvec3& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const uint32_t, nb::numpy, nb::shape<3>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  uint32_t(), nb::arg("other"))
    .def(nb::self += uint32_t(), nb::arg("other"))
    .def(uint32_t() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  uint32_t(), nb::arg("other"))
    .def(nb::self -= uint32_t(), nb::arg("other"))
    .def(uint32_t() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  uint32_t(), nb::arg("other"))
    .def(nb::self *= uint32_t(), nb::arg("other"))
    .def(uint32_t() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  uint32_t(), nb::arg("other"))
    .def(nb::self /= uint32_t(), nb::arg("other"))
    .def(uint32_t() /  nb::self, nb::arg("other"))
    .def(nb::self %  nb::self, nb::arg("other"))
    .def(nb::self %= nb::self, nb::arg("other"))
    .def(nb::self %  uint32_t(), nb::arg("other"))
    .def(nb::self %= uint32_t(), nb::arg("other"))
    .def(uint32_t() %  nb::self, nb::arg("other"))
    .def(nb::self &  nb::self, nb::arg("other"))
    .def(nb::self &= nb::self, nb::arg("other"))
    .def(nb::self &  uint32_t(), nb::arg("other"))
    .def(nb::self &= uint32_t(), nb::arg("other"))
    .def(uint32_t() &  nb::self, nb::arg("other"))
    .def(nb::self |  nb::self, nb::arg("other"))
    .def(nb::self |= nb::self, nb::arg("other"))
    .def(nb::self |  uint32_t(), nb::arg("other"))
    .def(nb::self |= uint32_t(), nb::arg("other"))
    .def(uint32_t() |  nb::self, nb::arg("other"))
    .def(nb::self ^  nb::self, nb::arg("other"))
    .def(nb::self ^= nb::self, nb::arg("other"))
    .def(nb::self ^  uint32_t(), nb::arg("other"))
    .def(nb::self ^= uint32_t(), nb::arg("other"))
    .def(uint32_t() ^  nb::self, nb::arg("other"))
    .def(-nb::self)
    .def(~nb::self)
;
nb::implicitly_convertible<nb::tuple, uvec3>();
nb::implicitly_convertible<nb::list, uvec3>();

nb::class_<uvec4>(mod_math, "uvec4")
    .def(nb::init<>())
    .def(nb::init<uint32_t, uint32_t, uint32_t, uint32_t>(), nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<uint32_t, uvec3>(), nb::arg("x"), nb::arg("yzw"))
    .def(nb::init<uvec3, uint32_t>(), nb::arg("xy"), nb::arg("z"))
    .def(nb::init<uint32_t, uint32_t, uvec2>(), nb::arg("x"), nb::arg("y"), nb::arg("zw"))
    .def(nb::init<uint32_t, uvec2, uint32_t>(), nb::arg("x"), nb::arg("yz"), nb::arg("w"))
    .def(nb::init<uvec2, uint32_t, uint32_t>(), nb::arg("xy"), nb::arg("z"), nb::arg("w"))
    .def(nb::init<vec4>(), nb::arg("v"))
    .def(nb::init<dvec4>(), nb::arg("v"))
    .def(nb::init<ivec4>(), nb::arg("v"))
    .def(nb::init<uvec4>(), nb::arg("v"))
    .def("__init__", [](uvec4* v, nb::tuple t) {
        if (t.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to uvec4", t.size());
        }
        new (v) uvec4(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]));
    }, nb::arg("t"))
    .def("__init__", [](uvec4* v, nb::list l) {
        if (l.size() != 4) {
            nb::raise_type_error("Cannot convert tuple of length %zu to uvec4", l.size());
        }
        new (v) uvec4(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]));
    }, nb::arg("l"))
    .def_rw("x", &uvec4::x, nb::arg("val"))
    .def_rw("y", &uvec4::y, nb::arg("val"))
    .def_rw("z", &uvec4::z, nb::arg("val"))
    .def_rw("w", &uvec4::w, nb::arg("val"))
    .def("__repr__", [](const uvec4& v) {
        char buf[128];
        snprintf(buf, sizeof(buf), "uvec4(%d, %d, %d, %d)", v.x, v.y, v.z, v.w);
        return nb::str(buf);
    })
    .def("__iter__", [](const uvec4 &v) {
        return nb::make_iterator(nb::type<uvec4>(), "uvec4_iterator", &v.x, &v.x + 4);
    })
    .def("__array__", [] (const uvec4& v, nb::handle dtype, std::optional<bool> copy) {
        return nb::ndarray<const uint32_t, nb::numpy, nb::shape<4>>(&v.x);
    }, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  uint32_t(), nb::arg("other"))
    .def(nb::self += uint32_t(), nb::arg("other"))
    .def(uint32_t() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  uint32_t(), nb::arg("other"))
    .def(nb::self -= uint32_t(), nb::arg("other"))
    .def(uint32_t() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  uint32_t(), nb::arg("other"))
    .def(nb::self *= uint32_t(), nb::arg("other"))
    .def(uint32_t() *  nb::self, nb::arg("other"))
    .def(nb::self /  nb::self, nb::arg("other"))
    .def(nb::self /= nb::self, nb::arg("other"))
    .def(nb::self /  uint32_t(), nb::arg("other"))
    .def(nb::self /= uint32_t(), nb::arg("other"))
    .def(uint32_t() /  nb::self, nb::arg("other"))
    .def(nb::self %  nb::self, nb::arg("other"))
    .def(nb::self %= nb::self, nb::arg("other"))
    .def(nb::self %  uint32_t(), nb::arg("other"))
    .def(nb::self %= uint32_t(), nb::arg("other"))
    .def(uint32_t() %  nb::self, nb::arg("other"))
    .def(nb::self &  nb::self, nb::arg("other"))
    .def(nb::self &= nb::self, nb::arg("other"))
    .def(nb::self &  uint32_t(), nb::arg("other"))
    .def(nb::self &= uint32_t(), nb::arg("other"))
    .def(uint32_t() &  nb::self, nb::arg("other"))
    .def(nb::self |  nb::self, nb::arg("other"))
    .def(nb::self |= nb::self, nb::arg("other"))
    .def(nb::self |  uint32_t(), nb::arg("other"))
    .def(nb::self |= uint32_t(), nb::arg("other"))
    .def(uint32_t() |  nb::self, nb::arg("other"))
    .def(nb::self ^  nb::self, nb::arg("other"))
    .def(nb::self ^= nb::self, nb::arg("other"))
    .def(nb::self ^  uint32_t(), nb::arg("other"))
    .def(nb::self ^= uint32_t(), nb::arg("other"))
    .def(uint32_t() ^  nb::self, nb::arg("other"))
    .def(-nb::self)
    .def(~nb::self)
;
nb::implicitly_convertible<nb::tuple, uvec4>();
nb::implicitly_convertible<nb::list, uvec4>();

nb::class_<mat2>(mod_math, "mat2")
    .def(nb::init<>())
    .def(nb::init<float>(), nb::arg("diag"))
    .def(nb::init<vec2, vec2>(), nb::arg("c0"), nb::arg("c1"))
    .def(nb::init<float, float, float, float>(), nb::arg("m00"), nb::arg("m10"), nb::arg("m01"), nb::arg("m11"))
    .def(nb::init<mat2>(), nb::arg("m"))
    .def(nb::init<mat3>(), nb::arg("m"))
    .def(nb::init<mat4>(), nb::arg("m"))
    .def(nb::init<dmat2>(), nb::arg("m"))
    .def(nb::init<dmat3>(), nb::arg("m"))
    .def(nb::init<dmat4>(), nb::arg("m"))
    .def("__init__", [](mat2* m, nb::tuple t) {
        if (t.size() == 4) {
            new (m) mat2(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]));
        } else if (t.size() == 2) {
            new (m) mat2(nb::cast<vec2>(t[0]), nb::cast<vec2>(t[1]));
        } else {
            nb::raise_type_error("Cannot convert tuple of length %zu to mat2", t.size());
        }
    }, nb::arg("t"))
    .def("__init__", [](mat2* m, nb::list l) {
        if (l.size() == 4) {
            new (m) mat2(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]));
        } else if (l.size() == 2) {
            new (m) mat2(nb::cast<vec2>(l[0]), nb::cast<vec2>(l[1]));
        } else {
            nb::raise_type_error("Cannot convert list of length %zu to mat2", l.size());
        }
    }, nb::arg("t"))
    .def("__repr__", [](const mat2& m) {
        char buf[256];
        snprintf(buf, sizeof(buf), "mat2((%g, %g),(%g, %g))", m[0][0], m[0][1], m[1][0], m[1][1]);
        return nb::str(buf);
    })
    .def("__iter__", [](const mat2 &v) {
        return nb::make_iterator(nb::type<mat2>(), "mat2_iterator", &v[0][0], &v[0][0] + 4);
    })
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  float (), nb::arg("other"))
    .def(nb::self += float (), nb::arg("other"))
    .def(float () +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  float (), nb::arg("other"))
    .def(nb::self -= float (), nb::arg("other"))
    .def(float () -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  float (), nb::arg("other"))
    .def(nb::self *= float (), nb::arg("other"))
    .def(float () *  nb::self, nb::arg("other"))
    .def(-nb::self)
;
mod_math.def("transpose", [](const mat2& m) { return transpose(m); }, nb::arg("m"));
mod_math.def("inverse", [](const mat2& m) { return inverse(m); }, nb::arg("m"));
nb::implicitly_convertible<nb::tuple, mat2>();
nb::implicitly_convertible<nb::list, mat2>();
 
nb::class_<mat3>(mod_math, "mat3")
    .def(nb::init<>())
    .def(nb::init<float>(), nb::arg("diag"))
    .def(nb::init<vec3, vec3, vec3>(), nb::arg("c0"), nb::arg("c1"), nb::arg("c2"))
    .def(nb::init<float, float, float, float, float, float, float, float, float>(), nb::arg("m00"), nb::arg("m10"), nb::arg("m20"), nb::arg("m01"), nb::arg("m11"), nb::arg("m21"), nb::arg("m02"), nb::arg("m12"), nb::arg("m22"))
    .def(nb::init<mat2>(), nb::arg("m"))
    .def(nb::init<mat3>(), nb::arg("m"))
    .def(nb::init<mat4>(), nb::arg("m"))
    .def(nb::init<dmat2>(), nb::arg("m"))
    .def(nb::init<dmat3>(), nb::arg("m"))
    .def(nb::init<dmat4>(), nb::arg("m"))
    .def("__init__", [](mat3* m, nb::tuple t) {
        if (t.size() == 9) {
            new (m) mat3(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]), nb::cast<float>(t[4]), nb::cast<float>(t[5]), nb::cast<float>(t[6]), nb::cast<float>(t[7]), nb::cast<float>(t[8]));
        } else if (t.size() == 3) {
            new (m) mat3(nb::cast<vec3>(t[0]), nb::cast<vec3>(t[1]), nb::cast<vec3>(t[2]));
        } else {
            nb::raise_type_error("Cannot convert tuple of length %zu to mat3", t.size());
        }
    }, nb::arg("t"))
    .def("__init__", [](mat3* m, nb::list l) {
        if (l.size() == 9) {
            new (m) mat3(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]), nb::cast<float>(l[4]), nb::cast<float>(l[5]), nb::cast<float>(l[6]), nb::cast<float>(l[7]), nb::cast<float>(l[8]));
        } else if (l.size() == 3) {
            new (m) mat3(nb::cast<vec3>(l[0]), nb::cast<vec3>(l[1]), nb::cast<vec3>(l[2]));
        } else {
            nb::raise_type_error("Cannot convert list of length %zu to mat3", l.size());
        }
    }, nb::arg("t"))
    .def("__repr__", [](const mat3& m) {
        char buf[256];
        snprintf(buf, sizeof(buf), "mat3((%g, %g, %g),(%g, %g, %g),(%g, %g, %g))", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
        return nb::str(buf);
    })
    .def("__iter__", [](const mat3 &v) {
        return nb::make_iterator(nb::type<mat3>(), "mat3_iterator", &v[0][0], &v[0][0] + 9);
    })
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  float (), nb::arg("other"))
    .def(nb::self += float (), nb::arg("other"))
    .def(float () +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  float (), nb::arg("other"))
    .def(nb::self -= float (), nb::arg("other"))
    .def(float () -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  float (), nb::arg("other"))
    .def(nb::self *= float (), nb::arg("other"))
    .def(float () *  nb::self, nb::arg("other"))
    .def(-nb::self)
;
mod_math.def("transpose", [](const mat3& m) { return transpose(m); }, nb::arg("m"));
mod_math.def("inverse", [](const mat3& m) { return inverse(m); }, nb::arg("m"));
nb::implicitly_convertible<nb::tuple, mat3>();
nb::implicitly_convertible<nb::list, mat3>();
 
nb::class_<mat4>(mod_math, "mat4")
    .def(nb::init<>())
    .def(nb::init<float>(), nb::arg("diag"))
    .def(nb::init<vec4, vec4, vec4, vec4>(), nb::arg("c0"), nb::arg("c1"), nb::arg("c2"), nb::arg("c3"))
    .def(nb::init<float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float>(), nb::arg("m00"), nb::arg("m10"), nb::arg("m20"), nb::arg("m30"), nb::arg("m01"), nb::arg("m11"), nb::arg("m21"), nb::arg("m31"), nb::arg("m02"), nb::arg("m12"), nb::arg("m22"), nb::arg("m32"), nb::arg("m03"), nb::arg("m13"), nb::arg("m23"), nb::arg("m33"))
    .def(nb::init<mat2>(), nb::arg("m"))
    .def(nb::init<mat3>(), nb::arg("m"))
    .def(nb::init<mat4>(), nb::arg("m"))
    .def(nb::init<dmat2>(), nb::arg("m"))
    .def(nb::init<dmat3>(), nb::arg("m"))
    .def(nb::init<dmat4>(), nb::arg("m"))
    .def("__init__", [](mat4* m, nb::tuple t) {
        if (t.size() == 16) {
            new (m) mat4(nb::cast<float>(t[0]), nb::cast<float>(t[1]), nb::cast<float>(t[2]), nb::cast<float>(t[3]), nb::cast<float>(t[4]), nb::cast<float>(t[5]), nb::cast<float>(t[6]), nb::cast<float>(t[7]), nb::cast<float>(t[8]), nb::cast<float>(t[9]), nb::cast<float>(t[10]), nb::cast<float>(t[11]), nb::cast<float>(t[12]), nb::cast<float>(t[13]), nb::cast<float>(t[14]), nb::cast<float>(t[15]));
        } else if (t.size() == 4) {
            new (m) mat4(nb::cast<vec4>(t[0]), nb::cast<vec4>(t[1]), nb::cast<vec4>(t[2]), nb::cast<vec4>(t[3]));
        } else {
            nb::raise_type_error("Cannot convert tuple of length %zu to mat4", t.size());
        }
    }, nb::arg("t"))
    .def("__init__", [](mat4* m, nb::list l) {
        if (l.size() == 16) {
            new (m) mat4(nb::cast<float>(l[0]), nb::cast<float>(l[1]), nb::cast<float>(l[2]), nb::cast<float>(l[3]), nb::cast<float>(l[4]), nb::cast<float>(l[5]), nb::cast<float>(l[6]), nb::cast<float>(l[7]), nb::cast<float>(l[8]), nb::cast<float>(l[9]), nb::cast<float>(l[10]), nb::cast<float>(l[11]), nb::cast<float>(l[12]), nb::cast<float>(l[13]), nb::cast<float>(l[14]), nb::cast<float>(l[15]));
        } else if (l.size() == 4) {
            new (m) mat4(nb::cast<vec4>(l[0]), nb::cast<vec4>(l[1]), nb::cast<vec4>(l[2]), nb::cast<vec4>(l[3]));
        } else {
            nb::raise_type_error("Cannot convert list of length %zu to mat4", l.size());
        }
    }, nb::arg("t"))
    .def("__repr__", [](const mat4& m) {
        char buf[256];
        snprintf(buf, sizeof(buf), "mat4((%g, %g, %g, %g),(%g, %g, %g, %g),(%g, %g, %g, %g),(%g, %g, %g, %g))", m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]);
        return nb::str(buf);
    })
    .def("__iter__", [](const mat4 &v) {
        return nb::make_iterator(nb::type<mat4>(), "mat4_iterator", &v[0][0], &v[0][0] + 16);
    })
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  float (), nb::arg("other"))
    .def(nb::self += float (), nb::arg("other"))
    .def(float () +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  float (), nb::arg("other"))
    .def(nb::self -= float (), nb::arg("other"))
    .def(float () -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  float (), nb::arg("other"))
    .def(nb::self *= float (), nb::arg("other"))
    .def(float () *  nb::self, nb::arg("other"))
    .def(-nb::self)
;
mod_math.def("transpose", [](const mat4& m) { return transpose(m); }, nb::arg("m"));
mod_math.def("inverse", [](const mat4& m) { return inverse(m); }, nb::arg("m"));
nb::implicitly_convertible<nb::tuple, mat4>();
nb::implicitly_convertible<nb::list, mat4>();
 
nb::class_<dmat2>(mod_math, "dmat2")
    .def(nb::init<>())
    .def(nb::init<double>(), nb::arg("diag"))
    .def(nb::init<dvec2, dvec2>(), nb::arg("c0"), nb::arg("c1"))
    .def(nb::init<double, double, double, double>(), nb::arg("m00"), nb::arg("m10"), nb::arg("m01"), nb::arg("m11"))
    .def(nb::init<mat2>(), nb::arg("m"))
    .def(nb::init<mat3>(), nb::arg("m"))
    .def(nb::init<mat4>(), nb::arg("m"))
    .def(nb::init<dmat2>(), nb::arg("m"))
    .def(nb::init<dmat3>(), nb::arg("m"))
    .def(nb::init<dmat4>(), nb::arg("m"))
    .def("__init__", [](dmat2* m, nb::tuple t) {
        if (t.size() == 4) {
            new (m) dmat2(nb::cast<double>(t[0]), nb::cast<double>(t[1]), nb::cast<double>(t[2]), nb::cast<double>(t[3]));
        } else if (t.size() == 2) {
            new (m) dmat2(nb::cast<dvec2>(t[0]), nb::cast<dvec2>(t[1]));
        } else {
            nb::raise_type_error("Cannot convert tuple of length %zu to dmat2", t.size());
        }
    }, nb::arg("t"))
    .def("__init__", [](dmat2* m, nb::list l) {
        if (l.size() == 4) {
            new (m) dmat2(nb::cast<double>(l[0]), nb::cast<double>(l[1]), nb::cast<double>(l[2]), nb::cast<double>(l[3]));
        } else if (l.size() == 2) {
            new (m) dmat2(nb::cast<dvec2>(l[0]), nb::cast<dvec2>(l[1]));
        } else {
            nb::raise_type_error("Cannot convert list of length %zu to dmat2", l.size());
        }
    }, nb::arg("t"))
    .def("__repr__", [](const dmat2& m) {
        char buf[256];
        snprintf(buf, sizeof(buf), "dmat2((%g, %g),(%g, %g))", m[0][0], m[0][1], m[1][0], m[1][1]);
        return nb::str(buf);
    })
    .def("__iter__", [](const dmat2 &v) {
        return nb::make_iterator(nb::type<dmat2>(), "dmat2_iterator", &v[0][0], &v[0][0] + 4);
    })
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  double(), nb::arg("other"))
    .def(nb::self += double(), nb::arg("other"))
    .def(double() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  double(), nb::arg("other"))
    .def(nb::self -= double(), nb::arg("other"))
    .def(double() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  double(), nb::arg("other"))
    .def(nb::self *= double(), nb::arg("other"))
    .def(double() *  nb::self, nb::arg("other"))
    .def(-nb::self)
;
mod_math.def("transpose", [](const dmat2& m) { return transpose(m); }, nb::arg("m"));
mod_math.def("inverse", [](const dmat2& m) { return inverse(m); }, nb::arg("m"));
nb::implicitly_convertible<nb::tuple, dmat2>();
nb::implicitly_convertible<nb::list, dmat2>();
 
nb::class_<dmat3>(mod_math, "dmat3")
    .def(nb::init<>())
    .def(nb::init<double>(), nb::arg("diag"))
    .def(nb::init<dvec3, dvec3, dvec3>(), nb::arg("c0"), nb::arg("c1"), nb::arg("c2"))
    .def(nb::init<double, double, double, double, double, double, double, double, double>(), nb::arg("m00"), nb::arg("m10"), nb::arg("m20"), nb::arg("m01"), nb::arg("m11"), nb::arg("m21"), nb::arg("m02"), nb::arg("m12"), nb::arg("m22"))
    .def(nb::init<mat2>(), nb::arg("m"))
    .def(nb::init<mat3>(), nb::arg("m"))
    .def(nb::init<mat4>(), nb::arg("m"))
    .def(nb::init<dmat2>(), nb::arg("m"))
    .def(nb::init<dmat3>(), nb::arg("m"))
    .def(nb::init<dmat4>(), nb::arg("m"))
    .def("__init__", [](dmat3* m, nb::tuple t) {
        if (t.size() == 9) {
            new (m) dmat3(nb::cast<double>(t[0]), nb::cast<double>(t[1]), nb::cast<double>(t[2]), nb::cast<double>(t[3]), nb::cast<double>(t[4]), nb::cast<double>(t[5]), nb::cast<double>(t[6]), nb::cast<double>(t[7]), nb::cast<double>(t[8]));
        } else if (t.size() == 3) {
            new (m) dmat3(nb::cast<dvec3>(t[0]), nb::cast<dvec3>(t[1]), nb::cast<dvec3>(t[2]));
        } else {
            nb::raise_type_error("Cannot convert tuple of length %zu to dmat3", t.size());
        }
    }, nb::arg("t"))
    .def("__init__", [](dmat3* m, nb::list l) {
        if (l.size() == 9) {
            new (m) dmat3(nb::cast<double>(l[0]), nb::cast<double>(l[1]), nb::cast<double>(l[2]), nb::cast<double>(l[3]), nb::cast<double>(l[4]), nb::cast<double>(l[5]), nb::cast<double>(l[6]), nb::cast<double>(l[7]), nb::cast<double>(l[8]));
        } else if (l.size() == 3) {
            new (m) dmat3(nb::cast<dvec3>(l[0]), nb::cast<dvec3>(l[1]), nb::cast<dvec3>(l[2]));
        } else {
            nb::raise_type_error("Cannot convert list of length %zu to dmat3", l.size());
        }
    }, nb::arg("t"))
    .def("__repr__", [](const dmat3& m) {
        char buf[256];
        snprintf(buf, sizeof(buf), "dmat3((%g, %g, %g),(%g, %g, %g),(%g, %g, %g))", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
        return nb::str(buf);
    })
    .def("__iter__", [](const dmat3 &v) {
        return nb::make_iterator(nb::type<dmat3>(), "dmat3_iterator", &v[0][0], &v[0][0] + 9);
    })
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  double(), nb::arg("other"))
    .def(nb::self += double(), nb::arg("other"))
    .def(double() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  double(), nb::arg("other"))
    .def(nb::self -= double(), nb::arg("other"))
    .def(double() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  double(), nb::arg("other"))
    .def(nb::self *= double(), nb::arg("other"))
    .def(double() *  nb::self, nb::arg("other"))
    .def(-nb::self)
;
mod_math.def("transpose", [](const dmat3& m) { return transpose(m); }, nb::arg("m"));
mod_math.def("inverse", [](const dmat3& m) { return inverse(m); }, nb::arg("m"));
nb::implicitly_convertible<nb::tuple, dmat3>();
nb::implicitly_convertible<nb::list, dmat3>();
 
nb::class_<dmat4>(mod_math, "dmat4")
    .def(nb::init<>())
    .def(nb::init<double>(), nb::arg("diag"))
    .def(nb::init<dvec4, dvec4, dvec4, dvec4>(), nb::arg("c0"), nb::arg("c1"), nb::arg("c2"), nb::arg("c3"))
    .def(nb::init<double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double>(), nb::arg("m00"), nb::arg("m10"), nb::arg("m20"), nb::arg("m30"), nb::arg("m01"), nb::arg("m11"), nb::arg("m21"), nb::arg("m31"), nb::arg("m02"), nb::arg("m12"), nb::arg("m22"), nb::arg("m32"), nb::arg("m03"), nb::arg("m13"), nb::arg("m23"), nb::arg("m33"))
    .def(nb::init<mat2>(), nb::arg("m"))
    .def(nb::init<mat3>(), nb::arg("m"))
    .def(nb::init<mat4>(), nb::arg("m"))
    .def(nb::init<dmat2>(), nb::arg("m"))
    .def(nb::init<dmat3>(), nb::arg("m"))
    .def(nb::init<dmat4>(), nb::arg("m"))
    .def("__init__", [](dmat4* m, nb::tuple t) {
        if (t.size() == 16) {
            new (m) dmat4(nb::cast<double>(t[0]), nb::cast<double>(t[1]), nb::cast<double>(t[2]), nb::cast<double>(t[3]), nb::cast<double>(t[4]), nb::cast<double>(t[5]), nb::cast<double>(t[6]), nb::cast<double>(t[7]), nb::cast<double>(t[8]), nb::cast<double>(t[9]), nb::cast<double>(t[10]), nb::cast<double>(t[11]), nb::cast<double>(t[12]), nb::cast<double>(t[13]), nb::cast<double>(t[14]), nb::cast<double>(t[15]));
        } else if (t.size() == 4) {
            new (m) dmat4(nb::cast<dvec4>(t[0]), nb::cast<dvec4>(t[1]), nb::cast<dvec4>(t[2]), nb::cast<dvec4>(t[3]));
        } else {
            nb::raise_type_error("Cannot convert tuple of length %zu to dmat4", t.size());
        }
    }, nb::arg("t"))
    .def("__init__", [](dmat4* m, nb::list l) {
        if (l.size() == 16) {
            new (m) dmat4(nb::cast<double>(l[0]), nb::cast<double>(l[1]), nb::cast<double>(l[2]), nb::cast<double>(l[3]), nb::cast<double>(l[4]), nb::cast<double>(l[5]), nb::cast<double>(l[6]), nb::cast<double>(l[7]), nb::cast<double>(l[8]), nb::cast<double>(l[9]), nb::cast<double>(l[10]), nb::cast<double>(l[11]), nb::cast<double>(l[12]), nb::cast<double>(l[13]), nb::cast<double>(l[14]), nb::cast<double>(l[15]));
        } else if (l.size() == 4) {
            new (m) dmat4(nb::cast<dvec4>(l[0]), nb::cast<dvec4>(l[1]), nb::cast<dvec4>(l[2]), nb::cast<dvec4>(l[3]));
        } else {
            nb::raise_type_error("Cannot convert list of length %zu to dmat4", l.size());
        }
    }, nb::arg("t"))
    .def("__repr__", [](const dmat4& m) {
        char buf[256];
        snprintf(buf, sizeof(buf), "dmat4((%g, %g, %g, %g),(%g, %g, %g, %g),(%g, %g, %g, %g),(%g, %g, %g, %g))", m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]);
        return nb::str(buf);
    })
    .def("__iter__", [](const dmat4 &v) {
        return nb::make_iterator(nb::type<dmat4>(), "dmat4_iterator", &v[0][0], &v[0][0] + 16);
    })
    .def(nb::self +  nb::self, nb::arg("other"))
    .def(nb::self += nb::self, nb::arg("other"))
    .def(nb::self +  double(), nb::arg("other"))
    .def(nb::self += double(), nb::arg("other"))
    .def(double() +  nb::self, nb::arg("other"))
    .def(nb::self -  nb::self, nb::arg("other"))
    .def(nb::self -= nb::self, nb::arg("other"))
    .def(nb::self -  double(), nb::arg("other"))
    .def(nb::self -= double(), nb::arg("other"))
    .def(double() -  nb::self, nb::arg("other"))
    .def(nb::self *  nb::self, nb::arg("other"))
    .def(nb::self *= nb::self, nb::arg("other"))
    .def(nb::self *  double(), nb::arg("other"))
    .def(nb::self *= double(), nb::arg("other"))
    .def(double() *  nb::self, nb::arg("other"))
    .def(-nb::self)
;
mod_math.def("transpose", [](const dmat4& m) { return transpose(m); }, nb::arg("m"));
mod_math.def("inverse", [](const dmat4& m) { return inverse(m); }, nb::arg("m"));
nb::implicitly_convertible<nb::tuple, dmat4>();
nb::implicitly_convertible<nb::list, dmat4>();
 

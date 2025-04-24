import os

out_cpp_file = open(os.path.join(os.path.dirname(__file__), "..", "src", "python", "generated_math.inc"), "w")

def out(*args, **kwargs):
    print(*args, **kwargs, file=out_cpp_file)

types = [
    ("", "float", False),
    ("d", "double", False),
    ("i", "int32_t", True),
    ("u", "uint32_t", True),
]

letters = [
    "x",
    "y",
    "z",
    "w",
]

common_unary_ops = [
    "-"
]

common_binary_ops = [
    "+",
    "-",
    "*",
    "/",
]

int_only_unary_ops = [
    "~"
]
int_only_binary_ops = [
    "%",
    "&",
    "|",
    "^",
]

float_unary_funcs = [
    "length",
    "length2",
    "normalize",
]

float_binary_funcs = [
    "dot",
    "distance",
    "distance2",
]

float3_binary_funcs = [
    "cross",
]

for pref, typ, is_int in types:
    for n in range(2, 5):
        # Constructors
        vec = f'{pref}vec{n}'
        out(f'nb::class_<{vec}>(mod_math, "{vec}")')
        out(" " * 4 + f'.def(nb::init<>())')
        nb_args = ", ".join([f'nb::arg("{l}")' for l in letters[:n]])
        out(" " * 4 + f'.def(nb::init<{", ".join([typ] * n)}>(), {nb_args})')
        if n == 3:
            vec2 = f'{pref}vec2'
            out(" " * 4 + f'.def(nb::init<{typ}, {vec2}>(), nb::arg("x"), nb::arg("yz"))')
            out(" " * 4 + f'.def(nb::init<{vec2}, {typ}>(), nb::arg("xy"), nb::arg("z"))')
        if n == 4:
            vec3 = f'{pref}vec3'
            out(" " * 4 + f'.def(nb::init<{typ}, {vec3}>(), nb::arg("x"), nb::arg("yzw"))')
            out(" " * 4 + f'.def(nb::init<{vec3}, {typ}>(), nb::arg("xy"), nb::arg("z"))')
            vec2 = f'{pref}vec2'
            out(" " * 4 + f'.def(nb::init<{typ}, {typ}, {vec2}>(), nb::arg("x"), nb::arg("y"), nb::arg("zw"))')
            out(" " * 4 + f'.def(nb::init<{typ}, {vec2}, {typ}>(), nb::arg("x"), nb::arg("yz"), nb::arg("w"))')
            out(" " * 4 + f'.def(nb::init<{vec2}, {typ}, {typ}>(), nb::arg("xy"), nb::arg("z"), nb::arg("w"))')
        for other_pref, other_typ, _ in types:
            for o in range(n, 5):
                ovec = f'{other_pref}vec{o}'
                out(" " * 4 + f'.def(nb::init<{ovec}>(), nb::arg("v"))')
        # List and tuple
        out(f'    .def("__init__", []({vec}* v, nb::tuple t) {{')
        out(f'        if (t.size() != {n}) {{')
        out(f'            nb::raise_type_error("Cannot convert tuple of length %zu to {vec}", t.size());')
        out(f'        }}')
        tuple_args = ", ".join([f"nb::cast<float>(t[{i}])" for i in range(n)])
        out(f'        new (v) {vec}({tuple_args});')
        out(f'    }}, nb::arg("t"))')
        out(f'    .def("__init__", []({vec}* v, nb::list l) {{')
        out(f'        if (l.size() != {n}) {{')
        out(f'            nb::raise_type_error("Cannot convert tuple of length %zu to {vec}", l.size());')
        out(f'        }}')
        list_args = ", ".join([f"nb::cast<float>(l[{i}])" for i in range(n)])
        out(f'        new (v) {vec}({list_args});')
        out(f'    }}, nb::arg("l"))')

        # Getters and setters
        for l in letters[:n]:
            out(" " * 4 + f'.def_rw("{l}", &{vec}::{l}, nb::arg("val"))')
        
        # Repr
        out(f'    .def("__repr__", [](const {vec}& v) {{')
        out(f'        char buf[128];')
        format_str = ", ".join(["%d" if is_int else "%g" for _ in range(n)])
        format_args = ", ".join([f"v.{l}" for l in letters[:n]])
        out(f'        snprintf(buf, sizeof(buf), "{vec}({format_str})", {format_args});')
        out(f'        return nb::str(buf);')
        out(f'    }})')

        # Iter
        out(f'    .def("__iter__", [](const {vec} &v) {{')
        out(f'        return nb::make_iterator(nb::type<{vec}>(), "{vec}_iterator", &v.x, &v.x + {n});')
        out(f'    }})')

        # Numpy interop
        out(f'    .def("__array__", [] (const {vec}& v, nb::handle dtype, std::optional<bool> copy) {{')
        out(f'        return nb::ndarray<const {typ}, nb::numpy, nb::shape<{n}>>(&v.x);')
        out(f'    }}, nb::rv_policy::copy, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())')

        # Operators
        binary_ops = list(common_binary_ops)
        if is_int:
            binary_ops += int_only_binary_ops
        for op in binary_ops:
            out(" " * 4 + f'.def(nb::self {op}  nb::self, nb::arg("other"))')
            out(" " * 4 + f'.def(nb::self {op}= nb::self, nb::arg("other"))')
            out(" " * 4 + f'.def(nb::self {op}  {typ:6}(), nb::arg("other"))')
            out(" " * 4 + f'.def(nb::self {op}= {typ:6}(), nb::arg("other"))')
            out(" " * 4 + f'.def({typ:6}() {op}  nb::self, nb::arg("other"))')
        unary_ops = list(common_unary_ops)
        if is_int:
            unary_ops += int_only_unary_ops
        for op in unary_ops:
            out(" " * 4 + f'.def({op}nb::self)')
        out(";")

        # List and tuple implicit conversion
        out(f"nb::implicitly_convertible<nb::tuple, {vec}>();")
        out(f"nb::implicitly_convertible<nb::list, {vec}>();")

        # Functions
        if not is_int:
            for f in float_unary_funcs:
                out(f'mod_math.def("{f}", [](const {vec}& v) {{ return {f}(v); }}, nb::arg("v"));')
            
            binary_funcs = list(float_binary_funcs)
            if n == 3:
                binary_funcs += float3_binary_funcs
            
            for f in binary_funcs:
                out(f'mod_math.def("{f}", [](const {vec}& a, const {vec}& b) {{ return {f}(a, b); }}, nb::arg("a"), nb::arg("b"));')


        out("")


import os
import itertools

out_cpp_file = open(os.path.join(os.path.dirname(__file__), "..", "src", "python", "generated_math.inc"), "w")

def out(*args, **kwargs):
    print(*args, **kwargs, file=out_cpp_file)

N_MIN = 2
N_MAX = 4

def gen_vec():
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
        for n in range(N_MIN, N_MAX + 1):
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
                for o in range(n, N_MAX + 1):
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

            # Accessors
            out(f'    .def("__getitem__", [](const {vec} &v, size_t i) {{')
            out(f'        if (i >= {n}) throw nb::index_error();')
            out(f'        return v[i];')
            out(f'    }})')

            out(f'    .def("__setitem__", []({vec} &v, size_t i, {typ} value) {{')
            out(f'        if (i >= {n}) throw nb::index_error();')
            out(f'        v[i] = value;')
            out(f'    }})')

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

def gen_mat():
    types = [
        ("", "float", False),
        ("d", "double", False),
    ]

    unary_ops = [
       "-", 
    ]

    binary_ops = [
       "+", 
       "-", 
       "*", 
    ]

    unary_funcs = [
        "transpose",
        "inverse",
    ]

    binary_funcs = [
    ]

    for pref, typ, is_int in types:
        for n in range(N_MIN, N_MAX + 1):
            # Constructors
            mat = f'{pref}mat{n}'
            colvec = f'{pref}vec{n}'
            out(f'nb::class_<{mat}>(mod_math, "{mat}")')
            out(" " * 4 + f'.def(nb::init<>())')
            out(" " * 4 + f'.def(nb::init<{typ}>(), nb::arg("diag"))')

            # From cols
            col_args = ", ".join([f'nb::arg("c{col}")' for col in range(n)])
            out(" " * 4 + f'.def(nb::init<{", ".join([colvec] * n)}>(), {col_args})')

            # From scalars
            scalar_args = ", ".join([f'nb::arg("m{row}{col}")' for col, row in itertools.product(range(n), range(n))])
            out(" " * 4 + f'.def(nb::init<{", ".join([typ] * n * n)}>(), {scalar_args})')
            for other_pref, _, _ in types:
                for o in range(N_MIN, N_MAX + 1):
                    omat = f'{other_pref}mat{o}'
                    out(" " * 4 + f'.def(nb::init<{omat}>(), nb::arg("m"))')

            # List and tuple
            tuple_scalar_args = ", ".join([f"nb::cast<{typ}>(t[{i}])" for i in range(n * n)])
            tuple_vector_args = ", ".join([f"nb::cast<{colvec}>(t[{i}])" for i in range(n)])
            out(f'    .def("__init__", []({mat}* m, nb::tuple t) {{')
            out(f'        if (t.size() == {n * n}) {{')
            out(f'            new (m) {mat}({tuple_scalar_args});')
            out(f'        }} else if (t.size() == {n}) {{')
            out(f'            new (m) {mat}({tuple_vector_args});')
            out(f'        }} else {{')
            out(f'            nb::raise_type_error("Cannot convert tuple of length %zu to {mat}", t.size());')
            out(f'        }}')
            out(f'    }}, nb::arg("t"))')
            list_scalar_args = ", ".join([f"nb::cast<{typ}>(l[{i}])" for i in range(n * n)])
            list_vector_args = ", ".join([f"nb::cast<{colvec}>(l[{i}])" for i in range(n)])
            out(f'    .def("__init__", []({mat}* m, nb::list l) {{')
            out(f'        if (l.size() == {n * n}) {{')
            out(f'            new (m) {mat}({list_scalar_args});')
            out(f'        }} else if (l.size() == {n}) {{')
            out(f'            new (m) {mat}({list_vector_args});')
            out(f'        }} else {{')
            out(f'            nb::raise_type_error("Cannot convert list of length %zu to {mat}", l.size());')
            out(f'        }}')
            out(f'    }}, nb::arg("t"))')

            # Repr
            out(f'    .def("__repr__", [](const {mat}& m) {{')
            out(f'        char buf[256];')
            format_str = ",".join([("(" + ", ".join(["%d" if is_int else "%g" for _ in range(n)]) + ")" ) for _ in range(n)])
            format_args = ", ".join([f"m[{col}][{row}]" for col, row in itertools.product(range(n), range(n))])
            out(f'        snprintf(buf, sizeof(buf), "{mat}({format_str})", {format_args});')
            out(f'        return nb::str(buf);')
            out(f'    }})')

            # Iter
            out(f'    .def("__iter__", [](const {mat} &v) {{')
            out(f'        return nb::make_iterator(nb::type<{mat}>(), "{mat}_iterator", &v[0][0], &v[0][0] + {n * n});')
            out(f'    }})')

            # Accessors
            out(f'    .def("__getitem__", [](const {mat} &m, size_t i) {{')
            out(f'        if (i >= {n}) throw nb::index_error();')
            out(f'        return m[i];')
            out(f'    }})')

            out(f'    .def("__setitem__", []({mat} &m, size_t i, {colvec} value) {{')
            out(f'        if (i >= {n}) throw nb::index_error();')
            out(f'        m[i] = value;')
            out(f'    }})')

            # out(f'    .def("__getitem__", [](const {mat} &m, std::array<size_t, {n}> i) {{')
            # out(f'        return m[i[0]][i[0]];')
            # out(f'    }})')

            # out(f'    .def("__setitem__", []({mat} &m, std::array<size_t, {n}>, {colvec} value) {{')
            # out(f'        if (i >= {n}) throw nb::index_error();')
            # out(f'        m[i] = value;')
            # out(f'    }})')

            # Numpy interop
            #
            # We transpose to use the c_contig order, which is the most standard and least surprising.
            #
            # This is not strictly necessary, we could specify the c_order here, but this could lead to
            # some inconsistencies between structurAed dtypes (that force c_order and would convert automatically)
            # and manual call to tobytes(). Since this copy should be pretty cheap we pick the more conventional option.
            #
            # We also need to nb::cast because we are referencing a temporary.
            # See: https://nanobind.readthedocs.io/en/latest/ndarray.html#returning-temporaries
            out(f'    .def("__array__", [] (const {mat}& m, nb::handle dtype, std::optional<bool> copy) {{')
            out(f'        {mat} t = transpose(m);')
            out(f'        return nb::cast(nb::ndarray<const {typ}, nb::numpy, nb::shape<{n},{n}>>(&t[0][0]));')
            out(f'    }}, nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())')

            # Operators
            for op in binary_ops:
                out(" " * 4 + f'.def(nb::self {op}  nb::self, nb::arg("other"))')
                out(" " * 4 + f'.def(nb::self {op}= nb::self, nb::arg("other"))')
                out(" " * 4 + f'.def(nb::self {op}  {typ:6}(), nb::arg("other"))')
                out(" " * 4 + f'.def(nb::self {op}= {typ:6}(), nb::arg("other"))')
                out(" " * 4 + f'.def({typ:6}() {op}  nb::self, nb::arg("other"))')
            for op in unary_ops:
                out(" " * 4 + f'.def({op}nb::self)')
            out(";")

            for f in unary_funcs:
                out(f'mod_math.def("{f}", [](const {mat}& m) {{ return {f}(m); }}, nb::arg("m"));')
                
            for f in binary_funcs:
                out(f'mod_math.def("{f}", [](const {mat}& a, const {mat}& b) {{ return {f}(a, b); }}, nb::arg("a"), nb::arg("b"));')

            # List and tuple implicit conversion
            out(f"nb::implicitly_convertible<nb::tuple, {mat}>();")
            out(f"nb::implicitly_convertible<nb::list, {mat}>();")

            out(" ")


gen_vec()
gen_mat()
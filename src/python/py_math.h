#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>                      // GLM math
#include <glm/gtc/matrix_transform.hpp>     // GLM matrix ops
#include <glm/gtx/norm.hpp>                 // length2, distance and distance2

// NB_MAKE_OPAQUE(glm::ivec2)

#if 0
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// TODO: likely can easily generalize to any N, T pairs, probably needs nested
// template, look at stl/array for reference.


template <typename Vec, typename Type, size_t Size>
struct vec_caster {
    NB_TYPE_CASTER(Vec, io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) +
                                const_name("[") + make_caster<Type>::Name +
                                const_name("]"))

    using Caster = make_caster<Type>;

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        PyObject *temp;

        /* Will initialize 'temp' (NULL in the case of a failure.) */
        PyObject **o = seq_get_with_size(src.ptr(), Size, &temp);

        bool success = o != nullptr;

        Caster caster;
        flags = flags_for_local_caster<int>(flags);

        if (success) {
            for (size_t i = 0; i < Size; ++i) {
                if (!caster.from_python(o[i], flags, cleanup) ||
                    !caster.template can_cast<Type>()) {
                    success = false;
                    break;
                }

                value[i] = caster.operator cast_t<Type>();
            }

            Py_XDECREF(temp);
        }

        return success;
    }

    static handle from_cpp(Vec&& src, rv_policy policy, cleanup_list *cleanup) {
        object ret = steal(PyTuple_New(2));

        if (ret.is_valid()) {
            Py_ssize_t index = 0;

            for (size_t i = 0; i < Size; ++i) {
                handle h = Caster::from_cpp(src[i], policy, cleanup);

                if (!h.is_valid()) {
                    ret.reset();
                    break;
                }

                NB_TUPLE_SET_ITEM(ret.ptr(), index++, h.ptr());
            }
        }

        return ret.release();
    }
};

template<typename Type>
struct type_caster<glm::vec<2, Type, glm::defaultp>>
    : vec_caster<glm::vec<2, Type, glm::defaultp>, Type, 2> { };

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#endif
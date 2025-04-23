#include <nanobind/stl/function.h>
#include <xpg/function.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename Return, typename... Args>
struct type_caster<xpg::Function<Return(Args...)>> {
    using ReturnCaster = make_caster<
        std::conditional_t<std::is_void_v<Return>, void_type, Return>>;

    NB_TYPE_CASTER(xpg::Function <Return(Args...)>,
                   const_name(NB_TYPING_CALLABLE "[[") +
                       concat(make_caster<Args>::Name...) + const_name("], ") +
                       ReturnCaster::Name + const_name("]"))

    struct pyfunc_wrapper_t : pyfunc_wrapper {
        using pyfunc_wrapper::pyfunc_wrapper;

        Return operator()(Args... args) const {
            gil_scoped_acquire acq;
            return cast<Return>(handle(f)((forward_t<Args>) args...));
        }
    };

    bool from_python(handle src, uint8_t flags, cleanup_list *) noexcept {
        if (src.is_none())
            return flags & cast_flags::convert;

        if (!PyCallable_Check(src.ptr()))
            return false;

        value = pyfunc_wrapper_t(src.ptr());
        return true;
    }

    static handle from_cpp(const Value &value, rv_policy rvp,
                           cleanup_list *) noexcept {
        const pyfunc_wrapper_t *wrapper = value.template dynamicCastTo<pyfunc_wrapper_t>();
        if (wrapper)
        {
            return handle(wrapper->f).inc_ref();
        }

        if (rvp == rv_policy::none)
            return handle();

        if (!value)
            return none().release();

        return cpp_function(value).release();
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)


#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;
typedef size_t   usize;
typedef float    f32;
typedef double   f64;
#define ArrayCount(a) (sizeof((a)) / sizeof(*(a)))
#define ZeroAlloc(size) (calloc(1, size))
#define Free(p) free((p))
#define OutOfBounds(i) (assert(false))
#define OutOfSpace() (assert(false))

#ifdef _WIN32
#define AlignedAlloc(align, size) (_aligned_malloc((size), (align)))
#define AlignedFree(ptr) (_aligned_free((ptr)))
#else
#define AlignedAlloc(align, size) (aligned_alloc(Max((align), sizeof(void*)), AlignUp(size, Max((align), sizeof(void*)))))
#define AlignedFree(ptr) (free((ptr)))
#endif


template<typename T>
T Max(const T& a, const T& b) {
    return a < b ? b : a;
}

template<typename T>
T Min(const T& a, const T& b) {
    return a < b ? a : b;
}

template<typename T>
T Clamp(const T& v, const T& min, const T& max) {
    return Min(Max(v, min), max);
}

inline bool IsPow2NonZero(usize n) {
    return (n & (n - 1)) == 0;
}

inline bool IsPow2(usize n) {
    return n == 0 || IsPow2NonZero(n);
}

inline usize AlignDown(usize v, usize a) {
    assert(IsPow2NonZero(a));
    return v & ~(a - 1);
}

inline usize AlignUp(usize v, usize a) {
    assert(IsPow2NonZero(a));
    return (v + (a - 1)) & ~(a - 1);
}

inline usize DivCeil(usize v, usize a) {
    assert(IsPow2NonZero(a));
    return (v + a - 1) / a;
}


template <class T> struct RemoveReference        { using type = T; };
template <class T> struct RemoveReference<T&>    { using type = T; };
template <class T> struct RemoveReference<T&&>   { using type = T; };
template <class T> struct AddPointer             { using type = typename RemoveReference<T>::type*; };
template <class T> struct IsLValueReference     { static constexpr bool value = false; };
template <class T> struct IsLValueReference<T&> { static constexpr bool value = true; };

template <typename T> constexpr T&& move(T& value) { return static_cast<T&&>(value); }
template <bool B, class T = void> struct EnableIf {};
template <class T> struct EnableIf<true, T> { using type = T; };
template <typename T, typename U>   struct IsSame       { static constexpr bool value = false; };
template <typename T>               struct IsSame<T, T> { static constexpr bool value = true;  };

template <typename T> constexpr T&& forward(typename RemoveReference<T>::type& value) { return static_cast<T&&>(value); }
template <typename T> constexpr T&& forward(typename RemoveReference<T>::type&& value)
{
    static_assert(!IsLValueReference<T>::value, "Forward an rvalue as an lvalue is not allowed");
    return static_cast<T&&>(value);
}

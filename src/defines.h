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
#define internal static


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

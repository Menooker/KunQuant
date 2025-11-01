#include "f32x4.hpp"
#include "s32x4.hpp"

namespace kun_simd {

template <typename T1, typename T2>
T1 cast(T2 v);

// cast with limited range
template <typename T1, typename T2>
T1 fast_cast(T2 v);

template <>
INLINE vec_f32x4 cast(vec_s32x4 v) {
    return vcvtq_f32_s32(v.v);
}

template <>
INLINE vec_s32x4 cast(vec_f32x4 v) {
    return vcvtq_s32_f32(v.v);
}

template <>
INLINE vec_f32x4 fast_cast(vec_s32x4 v) {
    return cast<vec_f32x4>(v);
}

template <>
INLINE vec_s32x4 fast_cast(vec_f32x4 v) {
    return cast<vec_s32x4>(v);
}


} // namespace kun_simd
#include "f32x4.hpp"
#include "s32x4.hpp"
#include "f64x2.hpp"
#include "s64x2.hpp"

namespace kun_simd {

template <typename T1, typename T2>
T1 cast(T2 v);

// cast with limited range
template <typename T1, typename T2>
T1 fast_cast(T2 v);

template <>
INLINE vec_f32x4 bitcast(vec_s32x4 v) {
    return vreinterpretq_f32_s32(v.v);
}

template <>
INLINE vec_s32x4 bitcast(vec_f32x4 v) {
    return vreinterpretq_s32_f32(v.v);
}

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

template <>
INLINE vec_f64x2 bitcast(vec_s64x2 v) {
    return vreinterpretq_f64_s64(v.v);
}

template <>
INLINE vec_s64x2 bitcast(vec_f64x2 v) {
    return vreinterpretq_s64_f64(v.v);
}

template <>
INLINE vec_f64x2 cast(vec_s64x2 v) {
    return vcvtq_f64_s64(v.v);
}

template <>
INLINE vec_s64x2 cast(vec_f64x2 v) {
    return vcvtq_s64_f64(v.v);
}

template <>
INLINE vec_f64x2 fast_cast(vec_s64x2 v) {
    return cast<vec_f64x2>(v);
}

template <>
INLINE vec_s64x2 fast_cast(vec_f64x2 v) {
    return cast<vec_s64x2>(v);
}


} // namespace kun_simd
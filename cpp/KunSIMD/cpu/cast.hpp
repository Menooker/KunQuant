#include "f32x8.hpp"
#include "f64x4.hpp"
#include "s32x8.hpp"
#include "s64x4.hpp"

namespace kun_simd {
template <typename T1, typename T2>
T1 bitcast(T2 v);

template <>
INLINE vec_f32x8 bitcast(vec_s32x8 v) {
    return _mm256_castsi256_ps(v.v);
}

template <>
INLINE vec_s32x8 bitcast(vec_f32x8 v) {
    return _mm256_castps_si256(v.v);
}

template <>
INLINE vec_f64x4 bitcast(vec_s64x4 v) {
    return _mm256_castsi256_pd(v.v);
}

template <>
INLINE vec_s64x4 bitcast(vec_f64x4 v) {
    return _mm256_castpd_si256(v.v);
}

template <typename T1, typename T2>
T1 cast(T2 v);

// cast with limited range
template <typename T1, typename T2>
T1 fast_cast(T2 v);

template <>
INLINE vec_f32x8 cast(vec_s32x8 v) {
    return _mm256_cvtepi32_ps(v.v);
}

template <>
INLINE vec_s32x8 cast(vec_f32x8 v) {
    return _mm256_cvtps_epi32(v.v);
}

template <>
INLINE vec_f32x8 fast_cast(vec_s32x8 v) {
    return cast<vec_f32x8>(v);
}

template <>
INLINE vec_s32x8 fast_cast(vec_f32x8 v) {
    return cast<vec_s32x8>(v);
}

#if !defined(__AVX512DQ__) || !defined(__AVX512VL__)

// https://stackoverflow.com/a/77376595
// Only works for inputs in the range: [-2^51, 2^51]
template <>
INLINE vec_s64x4 fast_cast(vec_f64x4 v) {
    auto x = _mm256_add_pd(v.v, _mm256_set1_pd(0x0018000000000000));
    return _mm256_sub_epi64(
        _mm256_castpd_si256(x),
        _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
}

// Only works for inputs in the range: [-2^51, 2^51]
template <>
INLINE vec_f64x4 fast_cast(vec_s64x4 v) {
    auto x = _mm256_add_epi64(
        v.v, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
    return _mm256_sub_pd(_mm256_castsi256_pd(x),
                         _mm256_set1_pd(0x0018000000000000));
}

template <>
INLINE vec_s64x4 cast(vec_f64x4 v) {
    const __m256d k2_32inv_dbl = _mm256_set1_pd(1.0 / 4294967296.0); // 1 / 2^32
    const __m256d k2_32_dbl = _mm256_set1_pd(4294967296.0);          // 2^32

    // Multiply by inverse instead of dividing.
    const __m256d v_hi_dbl = _mm256_mul_pd(v.v, k2_32inv_dbl);
    // Convert to integer.
    const vec_s64x4 v_hi = fast_cast<vec_s64x4>(vec_f64x4{v_hi_dbl});
    // Convert high32 integer to double and multiply by 2^32.
    const __m256d v_hi_int_dbl =
        _mm256_mul_pd(fast_cast<vec_f64x4>(v_hi), k2_32_dbl);
    // Subtract that from the original to get the remainder.
    const __m256d v_lo_dbl = _mm256_sub_pd(v.v, v_hi_int_dbl);
    // Convert to low32 integer.
    const __m256i v_lo = fast_cast<vec_s64x4>(vec_f64x4{v_lo_dbl});
    // Reconstruct integer from shifted high32 and remainder.
    return _mm256_add_epi64(_mm256_slli_epi64(v_hi.v, 32), v_lo);
}
#else
template <>
INLINE vec_s64x4 cast(vec_f64x4 v) {
    return _mm256_cvtpd_epi64(v.v);
}
template <>
INLINE vec_f64x4 cast(vec_s64x4 v) {
    return _mm256_cvtepi64_pd(v.v);
}

template <>
INLINE vec_s64x4 fast_cast(vec_f64x4 v) {
    return _mm256_cvtpd_epi64(v.v);
}
template <>
INLINE vec_f64x4 fast_cast(vec_s64x4 v) {
    return _mm256_cvtepi64_pd(v.v);
}
#endif

} // namespace kun_simd
#include "f32x16.hpp"
#include "f32x8.hpp"
#include "f64x4.hpp"
#include "f64x8.hpp"
#include "s32x16.hpp"
#include "s32x8.hpp"
#include "s64x4.hpp"
#include "s64x8.hpp"

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

#ifdef __AVX512F__

template <>
INLINE __mmask16 bitcast(__mmask16 v) {
    return v;
}

template <>
INLINE __mmask8 bitcast(__mmask8 v) {
    return v;
}

template <>
INLINE vec_f32x16 bitcast(vec_s32x16 v) {
    return _mm512_castsi512_ps(v.v);
}

template <>
INLINE vec_s32x16 bitcast(vec_f32x16 v) {
    return _mm512_castps_si512(v.v);
}

template <>
INLINE vec_f64x8 bitcast(vec_s64x8 v) {
    return _mm512_castsi512_pd(v.v);
}

template <>
INLINE vec_s64x8 bitcast(vec_f64x8 v) {
    return _mm512_castpd_si512(v.v);
}
#endif

template <typename T1, typename T2>
T1 cast(T2 v);

// cast with limited range
template <typename T1, typename T2>
T1 fast_cast(T2 v);

/////// start of f32 <==> s32
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

#ifdef __AVX512F__
template <>
INLINE vec_f32x16 cast(vec_s32x16 v) {
    return _mm512_cvtepi32_ps(v.v);
}

template <>
INLINE vec_s32x16 cast(vec_f32x16 v) {
    return _mm512_cvtps_epi32(v.v);
}

template <>
INLINE vec_f32x16 fast_cast(vec_s32x16 v) {
    return cast<vec_f32x16>(v);
}

template <>
INLINE vec_s32x16 fast_cast(vec_f32x16 v) {
    return cast<vec_s32x16>(v);
}
#endif

/////// end of f32 <==> s32

/////// start of f64x4 <==> s64x4
#if !defined(__AVX512DQ__) || !defined(__AVX512VL__)

// https://stackoverflow.com/a/77376595
// Only works for inputs in the range: [-2^51, 2^51]
template <>
INLINE vec_s64x4 fast_cast(vec_f64x4 v) {
    auto x = _mm256_add_pd(v.v, _mm256_set1_pd(0x0018000000000000));
    return vec_s64x4(_mm256_castpd_si256(x)) -
           _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000));
}

// Only works for inputs in the range: [-2^51, 2^51]
template <>
INLINE vec_f64x4 fast_cast(vec_s64x4 v) {
    auto x = v + _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000));
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

/////// end of f64x4 <==> s64x4

/////// end of f64x8 <==> s64x8 with AVX512DQ
#if defined(__AVX512F__) && defined(__AVX512DQ__)
template <>
INLINE vec_s64x8 cast(vec_f64x8 v) {
    return _mm512_cvtpd_epi64(v.v);
}
template <>
INLINE vec_f64x8 cast(vec_s64x8 v) {
    return _mm512_cvtepi64_pd(v.v);
}

template <>
INLINE vec_s64x8 fast_cast(vec_f64x8 v) {
    return _mm512_cvtpd_epi64(v.v);
}
template <>
INLINE vec_f64x8 fast_cast(vec_s64x8 v) {
    return _mm512_cvtepi64_pd(v.v);
}
#elif defined(__AVX512F__) && !defined(__AVX512DQ__)
template <>
INLINE vec_s64x8 fast_cast(vec_f64x8 v) {
    auto x = _mm512_add_pd(v.v, _mm512_set1_pd(0x0018000000000000));
    return _mm512_sub_epi64(
        _mm512_castpd_si512(x),
        _mm512_castpd_si512(_mm512_set1_pd(0x0018000000000000)));
}

// Only works for inputs in the range: [-2^51, 2^51]
template <>
INLINE vec_f64x8 fast_cast(vec_s64x8 v) {
    auto x = _mm512_add_epi64(
        v.v, _mm512_castpd_si512(_mm512_set1_pd(0x0018000000000000)));
    return _mm512_sub_pd(_mm512_castsi512_pd(x),
                         _mm512_set1_pd(0x0018000000000000));
}

template <>
INLINE vec_s64x8 cast(vec_f64x8 v) {
    const __m512d k2_32inv_dbl = _mm512_set1_pd(1.0 / 4294967296.0); // 1 / 2^32
    const __m512d k2_32_dbl = _mm512_set1_pd(4294967296.0);          // 2^32

    // Multiply by inverse instead of dividing.
    const __m512d v_hi_dbl = _mm512_mul_pd(v.v, k2_32inv_dbl);
    // Convert to integer.
    const vec_s64x8 v_hi = fast_cast<vec_s64x8>(vec_f64x8{v_hi_dbl});
    // Convert high32 integer to double and multiply by 2^32.
    const __m512d v_hi_int_dbl =
        _mm512_mul_pd(fast_cast<vec_f64x8>(v_hi), k2_32_dbl);
    // Subtract that from the original to get the remainder.
    const __m512d v_lo_dbl = _mm512_sub_pd(v.v, v_hi_int_dbl);
    // Convert to low32 integer.
    const __m512i v_lo = fast_cast<vec_s64x8>(vec_f64x8{v_lo_dbl});
    // Reconstruct integer from shifted high32 and remainder.
    return _mm512_add_epi64(_mm512_slli_epi64(v_hi.v, 32), v_lo);
}
#endif

INLINE vec_f32x8::Masktype vec_f32x8::make_mask(int N) {
    return bitcast<vec_f32x8::Masktype>(vec_s32x8{N} >
                                        vec_s32x8{0, 1, 2, 3, 4, 5, 6, 7});
}

INLINE vec_f64x4::Masktype vec_f64x4::make_mask(int N) {
    return bitcast<vec_f64x4::Masktype>(vec_s64x4{N} > vec_s64x4{0, 1, 2, 3});
}

} // namespace kun_simd
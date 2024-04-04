#include "f32x8.hpp"
#include "s32x8.hpp"
#include "f64x4.hpp"
#include "s64x4.hpp"

namespace kun_simd {
template<typename T1, typename T2>
T1 bitcast(T2 v);

template<>
INLINE vec_f32x8 bitcast(vec_s32x8 v) {
    return _mm256_castsi256_ps(v.v);
}

template<>
INLINE vec_s32x8 bitcast(vec_f32x8 v) {
    return _mm256_castps_si256(v.v);
}


template<>
INLINE vec_f64x4 bitcast(vec_s64x4 v) {
    return _mm256_castsi256_pd(v.v);
}


template<>
INLINE vec_s64x4 bitcast(vec_f64x4 v) {
    return _mm256_castpd_si256(v.v);
}

template<typename T1, typename T2>
T1 cast(T2 v);

template<>
INLINE vec_f32x8 cast(vec_s32x8 v) {
    return _mm256_cvtepi32_ps(v.v);
}

template<>
INLINE vec_s32x8 cast(vec_f32x8 v) {
    return _mm256_cvtps_epi32(v.v);
}

template<>
INLINE vec_f64x4 cast(vec_s64x4 v) {
    return _mm256_cvtepi64_pd(v.v);
}

template<>
INLINE vec_s64x4 cast(vec_f64x4 v) {
    return _mm256_cvtpd_epi64(v.v);
}


}
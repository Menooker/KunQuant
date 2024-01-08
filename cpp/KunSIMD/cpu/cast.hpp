#include "f32x8.hpp"
#include "s32x8.hpp"

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

}
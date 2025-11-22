#include "s32x16.hpp"
#include "s32x8.hpp"
#include "s64x4.hpp"
#include "s64x8.hpp"
#include "s32x4.hpp"
#include "s64x2.hpp"

namespace kun_simd {
#ifdef __AVX2__
template<int scale>
INLINE vec_f32x8 gather(const float* ptr, vec_s32x8 v) {
    return _mm256_i32gather_ps(ptr, v.v, scale);
}

template<int scale>
INLINE vec_f64x4 gather(const double* ptr, vec_s64x4 v) {
    return _mm256_i64gather_pd(ptr, v.v, scale);
}
template<int scale>
INLINE vec_f32x4 gather(const float* ptr, vec_s32x4 v) {
    return _mm_i32gather_ps(ptr, v.v, scale);
}

template<int scale>
INLINE vec_f32x4 gather(const double* ptr, vec_s32x4 v) {
    return _mm_i64gather_pd(ptr, v.v, scale);
}
#else
template<int scale>
INLINE vec_f32x4 gather(const float* ptr, vec_s32x4 v) {
    float out[4];
    for (int i = 0; i < 4; ++i) {
        out[i] = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(ptr) + v.raw[i] * scale);
    }
    return vec_f32x4::load(out);
}

// template<int scale>
// INLINE vec_f64x2 gather(const double* ptr, vec_s64x2 v) {
//     double out[2];
//     for (int i = 0; i < 2; ++i) {
//         out[i] = *reinterpret_cast<const double*>(reinterpret_cast<const char*>(ptr) + v.raw[i] * scale);
//     }
//     return vec_f32x4::load(out);
// }
#endif

#ifdef __AVX512F__
template<int scale>
INLINE vec_f32x16 gather(const float* ptr, vec_s32x16 v) {
    return _mm512_i32gather_ps(v.v, ptr, scale);
}

template<int scale>
INLINE vec_f64x8 gather(const double* ptr, vec_s64x8 v) {
    return _mm512_i64gather_pd(v.v, ptr, scale);
}

#endif

}
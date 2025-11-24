#include "s32x16.hpp"
#include "s32x8.hpp"
#include "s64x4.hpp"
#include "s64x8.hpp"

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
#else
template<int scale>
INLINE vec_f32x8 gather(const float* ptr, vec_s32x8 v) {
    float out[8];
    for (int i = 0; i < 8; ++i) {
        out[i] = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(ptr) + v.raw[i] * scale);
    }
    return vec_f32x8::load(out);
}

template<int scale>
INLINE vec_f64x4 gather(const double* ptr, vec_s64x4 v) {
    double out[4];
    for (int i = 0; i < 4; ++i) {
        out[i] = *reinterpret_cast<const double*>(reinterpret_cast<const char*>(ptr) + v.raw[i] * scale);
    }
    return vec_f64x4::load(out);
}
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
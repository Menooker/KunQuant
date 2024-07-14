#include "s32x16.hpp"
#include "s32x8.hpp"
#include "s64x4.hpp"
#include "s64x8.hpp"

namespace kun_simd {

template<int scale>
INLINE vec_f32x8 gather(const float* ptr, vec_s32x8 v) {
    return _mm256_i32gather_ps(ptr, v.v, scale);
}

template<int scale>
INLINE vec_f64x4 gather(const double* ptr, vec_s64x4 v) {
    return _mm256_i64gather_pd(ptr, v.v, scale);
}

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
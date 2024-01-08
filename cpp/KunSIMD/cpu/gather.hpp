#include "f32x8.hpp"
#include "s32x8.hpp"

namespace kun_simd {

template<int scale>
INLINE vec_f32x8 gather(const float* ptr, vec_s32x8 v) {
    return _mm256_i32gather_ps(ptr, v.v, scale);
}


}
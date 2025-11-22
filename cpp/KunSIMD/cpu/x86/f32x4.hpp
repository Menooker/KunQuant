/*******************************************************************************
 * SSE implementation for vec<float,4>
 *******************************************************************************/
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X4_HPP
#include <immintrin.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {

template <>
struct alignas(16) vec<float, 4> {
  public:
    union {
        __m128 v;
        float raw[4];
    };
    using Masktype = vec<float, 4>;

    using T = float;
    static constexpr int lanes = 4;

    INLINE vec() = default;
    INLINE vec(float f) { v = _mm_set1_ps(f); }
    INLINE vec(float i0, float i1, float i2, float i3) {
        v = _mm_setr_ps(i0, i1, i2, i3);
    }
    INLINE vec(__m128 const &x) { v = x; }

    static INLINE vec load(const float *p) { return _mm_loadu_ps(p); }
    static INLINE vec load_aligned(const float *p) { return _mm_load_ps(p); }
    static INLINE void store(vec v, float *p) { _mm_storeu_ps(p, v.v); }
    static INLINE void store_aligned(vec v, float *p) { _mm_store_ps(p, v.v); }
    static INLINE vec masked_load(const float *p, Masktype mask) {
        return _mm_maskload_ps(p, _mm_castps_si128(mask));
    }
    static INLINE void masked_store(vec v, float *p, Masktype mask) {
        _mm_maskstore_ps(p, _mm_castps_si128(mask), v.v);
    }

    static Masktype make_mask(int N);

    operator __m128() const { return v; }
};

using vec_f32x4 = vec<float, 4>;

INLINE vec_f32x4 operator+(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_add_ps(a.v, b.v);
}

INLINE vec_f32x4 operator-(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_sub_ps(a.v, b.v);
}

INLINE vec_f32x4 operator*(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_mul_ps(a.v, b.v);
}

INLINE vec_f32x4 operator/(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_div_ps(a.v, b.v);
}

INLINE vec_f32x4 sc_select(vec_f32x4 cond, vec_f32x4 const &a,
                           vec_f32x4 const &b) {
    return _mm_blendv_ps(b, a, cond);
}

INLINE vec_f32x4 operator==(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps(a.v, b.v, _CMP_EQ_OQ);
}
INLINE vec_f32x4 operator!=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps(a.v, b.v, _CMP_NEQ_OQ);
}
INLINE vec_f32x4 operator>(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps(a.v, b.v, _CMP_GT_OQ);
}
INLINE vec_f32x4 operator<(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps(a.v, b.v, _CMP_LT_OQ);
}
INLINE vec_f32x4 operator>=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps(a.v, b.v, _CMP_GE_OQ);
}
INLINE vec_f32x4 operator<=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_cmp_ps(a.v, b.v, _CMP_LE_OQ);
}

INLINE vec_f32x4 operator|(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_or_ps(a, b);
}
INLINE vec_f32x4 operator&(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_and_ps(a, b);
}
INLINE vec_f32x4 operator!(vec_f32x4 a) {
    return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1)));
}

INLINE vec_f32x4 sc_fmadd(vec_f32x4 const &a, vec_f32x4 const &b,
                          vec_f32x4 const &c) {
#ifdef __FMA__
    return _mm_fmadd_ps(a.v, b.v, c.v);
#else
    return _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v);
#endif
}

INLINE vec_f32x4 sc_fmsub(vec_f32x4 const &a, vec_f32x4 const &b,
                          vec_f32x4 const &c) {
#ifdef __FMA__
    return _mm_fmsub_ps(a.v, b.v, c.v);
#else
    return _mm_sub_ps(_mm_mul_ps(a.v, b.v), c.v);
#endif
}

INLINE vec_f32x4 sc_fnmadd(vec_f32x4 const &a, vec_f32x4 const &b,
                           vec_f32x4 const &c) {
#ifdef __FMA__
    return _mm_fnmadd_ps(a.v, b.v, c.v);
#else
    return _mm_sub_ps(c.v, _mm_mul_ps(a.v, b.v));
#endif
}

INLINE vec_f32x4 sc_max(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_max_ps(a.v, b.v);
}
INLINE vec_f32x4 sc_min(vec_f32x4 const &a, vec_f32x4 const &b) {
    return _mm_min_ps(a.v, b.v);
}

INLINE vec_f32x4 sc_round(vec_f32x4 const &a) {
    return _mm_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f32x4 sc_ceil(vec_f32x4 const &a) { return _mm_ceil_ps(a.v); }
INLINE vec_f32x4 sc_floor(vec_f32x4 const &a) { return _mm_floor_ps(a.v); }

INLINE vec_f32x4 sc_sqrt(vec_f32x4 const &a) { return _mm_sqrt_ps(a.v); }
INLINE vec_f32x4 sc_rsqrt(vec_f32x4 const &a) { return _mm_rsqrt_ps(a.v); }

INLINE vec_f32x4 sc_abs(vec_f32x4 const &a) {
    return _mm_andnot_ps(_mm_set1_ps(-0.0f), a.v);
}

INLINE vec_f32x4 sc_isnan(vec_f32x4 v1, vec_f32x4 v2) {
    return _mm_cmp_ps(v1.v, v2.v, _CMP_UNORD_Q);
}

INLINE vec_f32x4 sc_isnan(vec_f32x4 v1) {
    return _mm_cmp_ps(v1.v, v1.v, _CMP_UNORD_Q);
}

} // namespace kun_simd
#endif
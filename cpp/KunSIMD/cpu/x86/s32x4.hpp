/*******************************************************************************
 * SSE implementation for vec<int32_t,4>
 *******************************************************************************/
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S32X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S32X4_HPP
#include <immintrin.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {

template <>
struct alignas(16) vec<int32_t, 4> {
  public:
    union {
        __m128i v;
        int32_t raw[4];
    };
    using Masktype = vec<int32_t, 4>;

    using T = int32_t;
    static constexpr int lanes = 4;

    INLINE vec() = default;
    INLINE vec(int32_t i) { v = _mm_set1_epi32(i); }
    INLINE vec(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
        v = _mm_setr_epi32(i0, i1, i2, i3);
    }
    INLINE vec(__m128i const &x) { v = x; }

    static INLINE vec load(const int32_t *p) { return _mm_loadu_si128((const __m128i *)p); }
    static INLINE vec load_aligned(const int32_t *p) { return _mm_load_si128((const __m128i *)p); }
    static INLINE void store(vec v, int32_t *p) { _mm_storeu_si128((__m128i *)p, v.v); }
    static INLINE void store_aligned(vec v, int32_t *p) { _mm_store_si128((__m128i *)p, v.v); }
    static INLINE vec masked_load(const int32_t *p, Masktype mask) {
        return _mm_maskload_epi32(p, mask.v);
    }
    static INLINE void masked_store(vec v, int32_t *p, Masktype mask) {
        _mm_maskstore_epi32(p, mask.v, v.v);
    }

    operator __m128i() const { return v; }
};

using vec_s32x4 = vec<int32_t, 4>;

INLINE vec_s32x4 operator+(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_add_epi32(a.v, b.v);
}

INLINE vec_s32x4 operator-(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_sub_epi32(a.v, b.v);
}

INLINE vec_s32x4 operator*(vec_s32x4 const &a, vec_s32x4 const &b) {
#ifdef __SSE4_1__
    return _mm_mullo_epi32(a.v, b.v);
#else
    // fallback: multiply lanes using scalar extraction (less optimal)
    int32_t ra[4], rb[4];
    _mm_storeu_si128((__m128i *)ra, a.v);
    _mm_storeu_si128((__m128i *)rb, b.v);
    return vec_s32x4(ra[0] * rb[0], ra[1] * rb[1], ra[2] * rb[2], ra[3] * rb[3]);
#endif
}

INLINE vec_s32x4 sc_select(vec_s32x4 cond, vec_s32x4 const &a,
                           vec_s32x4 const &b) {
    __m128i res =
        _mm_blendv_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b),
                         _mm_castsi128_ps(mask));
    return _mm_castps_si128(res);
}

INLINE vec_s32x4 operator==(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmpeq_epi32(a.v, b.v);
}
INLINE vec_s32x4 operator!=(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_xor_si128(_mm_cmpeq_epi32(a.v, b.v), _mm_set1_epi32(-1));
}
INLINE vec_s32x4 operator>(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmpgt_epi32(a.v, b.v);
}
INLINE vec_s32x4 operator<(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmpgt_epi32(b.v, a.v);
}
INLINE vec_s32x4 operator>=(vec_s32x4 const &a, vec_s32x4 const &b) {
    // a >= b  <=> !(a < b)
    __m128i lt = _mm_cmpgt_epi32(b.v, a.v);
    return _mm_xor_si128(lt, _mm_set1_epi32(-1));
}
INLINE vec_s32x4 operator<=(vec_s32x4 const &a, vec_s32x4 const &b) {
    __m128i gt = _mm_cmpgt_epi32(a.v, b.v);
    return _mm_xor_si128(gt, _mm_set1_epi32(-1));
}

INLINE vec_s32x4 operator|(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_or_si128(a, b);
}
INLINE vec_s32x4 operator&(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_and_si128(a, b);
}
INLINE vec_s32x4 operator!(vec_s32x4 a) {
    return _mm_xor_si128(a, _mm_set1_epi32(-1));
}


INLINE vec_s32x8 operator<<(vec_s32x8 const &a, int64_t v) {
    return _mm_sll_epi32(a.v, b.v);
}
INLINE vec_s32x8 operator>>(vec_s32x8 const &a, int64_t v) {
    return _mm_sra_epi32(a.v, b.v);
}

INLINE vec_s32x8 logical_shr(vec_s32x8 const &a, int64_t v) {
    return _mm_srlv_epi32(a.v, b.v);
}

} // namespace kun_simd
#endif
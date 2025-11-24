#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S32X16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S32X16_HPP
#ifdef __AVX512F__
#include <immintrin.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>
namespace kun_simd {
template<>
struct alignas(64) vec<int32_t, 16> {
public:
    union {
        __m512i v;
        int32_t raw[16];
    };
    using Masktype = __mmask16;
    using T = int32_t;
    static constexpr int lanes = 16;

    INLINE vec() = default;
    INLINE vec(int32_t f) { v = _mm512_set1_epi32(f); }
    // INLINE vec(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4,
    //         int32_t i5, int32_t i6, int32_t i7) {
    //     v = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
    // }
    INLINE vec(__m512i const &x) { v = x; }
    // INLINE operator vec_f32x8() const;

    static INLINE vec load(const int32_t *p) {
        return _mm512_loadu_si512((const __m512i *)p);
    }
    static INLINE vec load_aligned(const int32_t *p) {
        return _mm512_load_si512((const __m512i *)p);
    }
    static INLINE void store(vec v, int32_t *p) {
        _mm512_storeu_si512((__m512i *)p, v.v);
    }
    static INLINE void store_aligned(vec v, int32_t *p) {
        _mm512_store_si512((__m512i *)p, v.v);
    }
    operator __m512i() const { return v; }
};

using vec_s32x16 = vec<int32_t, 16>;

INLINE vec_s32x16 operator+(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_add_epi32(a.v, b.v);
}

INLINE vec_s32x16 operator-(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_sub_epi32(a.v, b.v);
}
INLINE vec_s32x16 operator-(vec_s32x16 const &a) {
    return _mm512_sub_epi32(_mm512_setzero_si512(), a.v);
}

INLINE vec_s32x16 operator*(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_mullo_epi32(a.v, b.v);
}

// INLINE vec_s32x16 operator/(vec_s32x16 const &a, vec_s32x16 const &b) {
//     return _mm512_div_epi32(a.v, b.v);
// }

INLINE vec_s32x16 operator~(vec_s32x16 const &a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
}
INLINE vec_s32x16 operator&(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_and_si512(a.v, b.v);
}
INLINE vec_s32x16 operator|(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_or_si512(a.v, b.v);
}
INLINE vec_s32x16 operator^(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_xor_si512(a.v, b.v);
}

INLINE __mmask16 operator!(vec_s32x16 const &a) {
    return _mm512_cmp_epi32_mask(a.v, _mm512_setzero_si512(), _MM_CMPINT_EQ);
}
INLINE __mmask16 operator==(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask16 operator!=(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask16 operator>(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask16 operator<(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask16 operator>=(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask16 operator<=(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_s32x16 sc_select(
        __mmask16 mask, vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_mask_blend_epi32(mask, b.v, a.v);
}

INLINE vec_s32x16 operator<<(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_sllv_epi32(a.v, b.v);
}
INLINE vec_s32x16 operator>>(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_srav_epi32(a.v, b.v);
}

INLINE vec_s32x16 logical_shr(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_srlv_epi32(a.v, b.v);
}

// operator /

INLINE vec_s32x16 sc_max(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_max_epi32(a.v, b.v);
}
INLINE vec_s32x16 sc_min(vec_s32x16 const &a, vec_s32x16 const &b) {
    return _mm512_min_epi32(a.v, b.v);
}

INLINE vec_s32x16 sc_abs(vec_s32x16 const &a) {
    return _mm512_abs_epi32(a.v);
}

template <int v>
INLINE vec_s32x16 logical_shl(vec_s32x16 const &a) {
    return _mm512_slli_epi32(a.v, v);
}

template <int v>
INLINE vec_s32x16 logical_shr(vec_s32x16 const &a) {
    return _mm512_srli_epi32(a.v, v);
}

}
#endif
#endif
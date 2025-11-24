#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S64X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S64X8_HPP
#ifdef __AVX512F__
#include <immintrin.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>
namespace kun_simd {
template<>
struct alignas(64) vec<int64_t, 8> {
public:
    union {
        __m512i v;
        int64_t raw[8];
    };

    using Masktype = __mmask8;
    using T = int64_t;
    static constexpr int lanes = 8;

    INLINE vec() = default;
    INLINE vec(int64_t f) { v = _mm512_set1_epi64(f); }
    // INLINE vec(int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    //     v = _mm512_setr_epi64x(i0, i1, i2, i3);
    // }
    INLINE vec(__m512i const &x) { v = x; }
    // INLINE operator vec_f32x8() const;

    static INLINE vec load(const int64_t *p) {
        return _mm512_loadu_si512((const __m512i *)p);
    }
    static INLINE vec load_aligned(const int64_t *p) {
        return _mm512_load_si512((const __m512i *)p);
    }
    static INLINE void store(vec v, int64_t *p) {
        _mm512_storeu_si512((__m512i *)p, v.v);
    }
    static INLINE void store_aligned(vec v, int64_t *p) {
        _mm512_store_si512((__m512i *)p, v.v);
    }
    operator __m512i() const { return v; }
};

using vec_s64x8 = vec<int64_t, 8>;

INLINE vec_s64x8 operator+(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_add_epi64(a.v, b.v);
}

INLINE vec_s64x8 operator-(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_sub_epi64(a.v, b.v);
}
INLINE vec_s64x8 operator-(vec_s64x8 const &a) {
    return _mm512_sub_epi64(_mm512_setzero_si512(), a.v);
}

#if defined(__AVX512DQ__)

INLINE vec_s64x8 operator*(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_mullo_epi64(a.v, b.v);
}

INLINE vec_s64x8 sc_max(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_max_epi64(a.v, b.v);
}
INLINE vec_s64x8 sc_min(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_min_epi64(a.v, b.v);
}

INLINE vec_s64x8 sc_abs(vec_s64x8 const &a) {
    return _mm512_abs_epi64(a.v);
}

#endif

// INLINE vec_s64x8 operator/(vec_s64x8 const &a, vec_s64x8 const &b) {
//     return _mm512_div_epi64(a.v, b.v);
// }

INLINE vec_s64x8 operator~(vec_s64x8 const &a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi64(-1));
}
INLINE vec_s64x8 operator&(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_and_si512(a.v, b.v);
}
INLINE vec_s64x8 operator|(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_or_si512(a.v, b.v);
}
INLINE vec_s64x8 operator^(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_xor_si512(a.v, b.v);
}


INLINE __mmask8 operator==(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_EQ);
}

INLINE vec_s64x8 sc_select(
        __mmask8 mask, vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_mask_blend_epi64(mask, b.v, a.v);
}

INLINE vec_s64x8 operator<<(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_sllv_epi64(a.v, b.v);
}

INLINE vec_s64x8 operator>>(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_srav_epi64(a.v, b.v);
}

INLINE vec_s64x8 logical_shr(vec_s64x8 const &a, vec_s64x8 const &b) {
    return _mm512_srlv_epi64(a.v, b.v);
}

template <int v>
INLINE vec_s64x8 logical_shl(vec_s64x8 const &a) {
    return _mm512_slli_epi64(a.v, v);
}

template <int v>
INLINE vec_s64x8 logical_shr(vec_s64x8 const &a) {
    return _mm512_srli_epi64(a.v, v);
}

}
#endif
#endif
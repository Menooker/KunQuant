#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F64X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F64X8_HPP
#ifdef __AVX512F__
#include <immintrin.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {

template <>
struct alignas(64) vec<double, 8> {
  public:
    union {
        __m512d v;
        double raw[8];
    };

    using Masktype = __mmask8;
    using T = double;
    static constexpr int lanes = 8;

    INLINE vec() = default;
    INLINE vec(double f) { v = _mm512_set1_pd(f); }
    // INLINE vec(double i0, double i1, double i2, double i3) {
    //     v = _mm512_setr_pd(i0, i1, i2, i3);
    // }
    INLINE vec(__m512d const &x) { v = x; }

    static INLINE vec load(const double *p) { return _mm512_loadu_pd(p); }
    static INLINE vec load_aligned(const double *p) { return _mm512_load_pd(p); }
    static INLINE void store(vec v, double *p) { _mm512_storeu_pd(p, v.v); }
    static INLINE void store_aligned(vec v, double *p) {
        _mm512_store_pd(p, v.v);
    }
    static INLINE vec masked_load(const double *p, Masktype mask) {
        return _mm512_mask_loadu_pd(vec{0}, mask, p);
    }
    static INLINE void masked_store(vec v, double *p, Masktype mask) {
        _mm512_mask_storeu_pd(p, mask, v.v);
    }

    static INLINE Masktype make_mask(int N) {
        return (Masktype(1) << Masktype(N)) - Masktype(1);
    }
    operator __m512d() const { return v; }
};

using vec_f64x8 = vec<double, 8>;

INLINE vec_f64x8 operator+(vec_f64x8 const &a, vec_f64x8 const &b) {
    return _mm512_add_pd(a.v, b.v);
}

INLINE vec_f64x8 operator-(vec_f64x8 const &a, vec_f64x8 const &b) {
    return _mm512_sub_pd(a.v, b.v);
}

INLINE vec_f64x8 operator*(vec_f64x8 const &a, vec_f64x8 const &b) {
    return _mm512_mul_pd(a.v, b.v);
}

INLINE vec_f64x8 operator/(vec_f64x8 const &a, vec_f64x8 const &b) {
    return _mm512_div_pd(a.v, b.v);
}

INLINE vec_f64x8 sc_select(__mmask8 cond, vec_f64x8 const &a,
                           vec_f64x8 const &b) {
    return _mm512_mask_blend_pd(cond, b, a);
}

INLINE __mmask8 operator==(vec_f64x8 const &a, vec_f64x8 const &b) {
    auto ret = _mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ);
    return ret;
}
INLINE __mmask8 operator!=(vec_f64x8 const &a, vec_f64x8 const &b) {
    auto ret = _mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ);
    return ret;
}
INLINE __mmask8 operator>(vec_f64x8 const &a, vec_f64x8 const &b) {
    auto ret = _mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ);
    return ret;
}
INLINE __mmask8 operator<(vec_f64x8 const &a, vec_f64x8 const &b) {
    auto ret = _mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ);
    return ret;
}
INLINE __mmask8 operator>=(vec_f64x8 const &a, vec_f64x8 const &b) {
    auto ret = _mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ);
    return ret;
}
INLINE __mmask8 operator<=(vec_f64x8 const &a, vec_f64x8 const &b) {
    auto ret = _mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ);
    return ret;
}

// INLINE vec_f64x8 operator|(vec_f64x8 const &a, vec_f64x8 const &b) {
//     return _mm512_or_pd(a, b);
// }
// INLINE vec_f64x8 operator&(vec_f64x8 const &a, vec_f64x8 const &b) {
//     return _mm512_and_pd(a, b);
// }
// INLINE vec_f64x8 operator!(vec_f64x8 a) {
//     return _mm512_xor_pd(a, _mm512_castsi512_pd(_mm512_set1_epi64(-1)));
// }

INLINE vec_f64x8 sc_fmadd(vec_f64x8 const &a, vec_f64x8 const &b,
                          vec_f64x8 const &c) {
    return _mm512_fmadd_pd(a.v, b.v, c.v);
}

INLINE vec_f64x8 sc_fmsub(vec_f64x8 const &a, vec_f64x8 const &b,
                          vec_f64x8 const &c) {
    return _mm512_fmsub_pd(a.v, b.v, c.v);
}

INLINE vec_f64x8 sc_fnmadd(vec_f64x8 const &a, vec_f64x8 const &b,
                           vec_f64x8 const &c) {
    return _mm512_fnmadd_pd(a.v, b.v, c.v);
}

INLINE vec_f64x8 sc_max(vec_f64x8 const &a, vec_f64x8 const &b) {
    return _mm512_max_pd(a.v, b.v);
}
INLINE vec_f64x8 sc_min(vec_f64x8 const &a, vec_f64x8 const &b) {
    return _mm512_min_pd(a.v, b.v);
}

INLINE vec_f64x8 sc_round(vec_f64x8 const &a) {
    return _mm512_roundscale_pd(a.v, _MM_FROUND_TO_NEAREST_INT);
}

INLINE vec_f64x8 sc_ceil(vec_f64x8 const &a) { return _mm512_ceil_pd(a.v); }
INLINE vec_f64x8 sc_floor(vec_f64x8 const &a) { return _mm512_floor_pd(a.v); }

INLINE vec_f64x8 sc_sqrt(vec_f64x8 const &a) { return _mm512_sqrt_pd(a.v); }

INLINE vec_f64x8 sc_abs(vec_f64x8 const &a) {
    return _mm512_abs_pd(a.v);
}

inline __mmask8 sc_isnan(vec_f64x8 v1, vec_f64x8 v2) {
    return _mm512_cmp_pd_mask(v1.v, v2.v, _CMP_UNORD_Q);
}

inline __mmask8 sc_isnan(vec_f64x8 v1) {
    return _mm512_cmp_pd_mask(v1.v, v1.v, _CMP_UNORD_Q);
}

} // namespace kun_simd
#endif
#endif
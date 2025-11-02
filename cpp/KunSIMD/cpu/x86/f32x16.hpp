/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X16_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X16_HPP
#ifdef __AVX512F__
#include <immintrin.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {

template <>
struct alignas(64) vec<float, 16> {
  public:
    union {
        __m512 v;
        float raw[16];
    };

    using Masktype = __mmask16;
    using T = float;
    static constexpr int lanes = 16;

    INLINE vec() = default;
    INLINE vec(float f) { v = _mm512_set1_ps(f); }
    // INLINE vec(float i0, float i1, float i2, float i3, float i4, float i5,
    //            float i6, float i7) {
    //     v = _mm512_setr_ps(i0, i1, i2, i3, i4, i5, i6, i7);
    // }
    INLINE vec(__m512 const &x) { v = x; }

    static INLINE vec load(const float *p) { return _mm512_loadu_ps(p); }
    static INLINE vec load_aligned(const float *p) { return _mm512_load_ps(p); }
    static INLINE void store(vec v, float *p) { _mm512_storeu_ps(p, v.v); }
    static INLINE void store_aligned(vec v, float *p) {
        _mm512_store_ps(p, v.v);
    }
    static INLINE vec masked_load(const float *p, Masktype mask) {
        return _mm512_mask_loadu_ps(vec{0}, mask, p);
    }
    static INLINE void masked_store(vec v, float *p, Masktype mask) {
        _mm512_mask_storeu_ps(p, mask, v.v);
    }

    static INLINE Masktype make_mask(int N) {
        return (Masktype(1) << Masktype(N)) - Masktype(1);
    }
    operator __m512() const { return v; }
};

using vec_f32x16 = vec<float, 16>;

INLINE vec_f32x16 operator+(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_add_ps(a.v, b.v);
}

INLINE vec_f32x16 operator-(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_sub_ps(a.v, b.v);
}

INLINE vec_f32x16 operator*(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_mul_ps(a.v, b.v);
}

INLINE vec_f32x16 operator/(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_div_ps(a.v, b.v);
}

INLINE vec_f32x16 sc_select(__mmask16 cond, vec_f32x16 const &a,
                           vec_f32x16 const &b) {
  return _mm512_mask_blend_ps(cond, b.v, a.v);
}

INLINE __mmask16 operator==(vec_f32x16 const &a, vec_f32x16 const &b) {
    auto ret = _mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
    return ret;
}
INLINE __mmask16 operator!=(vec_f32x16 const &a, vec_f32x16 const &b) {
    auto ret = _mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ);
    return ret;
}
INLINE __mmask16 operator>(vec_f32x16 const &a, vec_f32x16 const &b) {
    auto ret = _mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
    return ret;
}
INLINE __mmask16 operator<(vec_f32x16 const &a, vec_f32x16 const &b) {
    auto ret = _mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
    return ret;
}
INLINE __mmask16 operator>=(vec_f32x16 const &a, vec_f32x16 const &b) {
    auto ret = _mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
    return ret;
}
INLINE __mmask16 operator<=(vec_f32x16 const &a, vec_f32x16 const &b) {
    auto ret = _mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
    return ret;
}

// INLINE vec_f32x16 operator|(vec_f32x16 const &a, vec_f32x16 const &b) {
//     return _mm512_or_ps(a, b);
// }
// INLINE vec_f32x16 operator&(vec_f32x16 const &a, vec_f32x16 const &b) {
//     return _mm512_and_ps(a, b);
// }
// INLINE vec_f32x16 operator!(vec_f32x16 a) {
//     return _mm512_xor_ps(a, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
// }

INLINE vec_f32x16 sc_fmadd(vec_f32x16 const &a, vec_f32x16 const &b,
                          vec_f32x16 const &c) {
    return _mm512_fmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x16 sc_fmsub(vec_f32x16 const &a, vec_f32x16 const &b,
                          vec_f32x16 const &c) {
    return _mm512_fmsub_ps(a.v, b.v, c.v);
}

INLINE vec_f32x16 sc_fnmadd(vec_f32x16 const &a, vec_f32x16 const &b,
                           vec_f32x16 const &c) {
    return _mm512_fnmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x16 sc_max(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_max_ps(a.v, b.v);
}
INLINE vec_f32x16 sc_min(vec_f32x16 const &a, vec_f32x16 const &b) {
    return _mm512_min_ps(a.v, b.v);
}

INLINE vec_f32x16 sc_round(vec_f32x16 const &a) {
  return _mm512_roundscale_ps(a.v, _MM_FROUND_TO_NEAREST_INT);
}

INLINE vec_f32x16 sc_ceil(vec_f32x16 const &a) { return _mm512_ceil_ps(a.v); }
INLINE vec_f32x16 sc_floor(vec_f32x16 const &a) { return _mm512_floor_ps(a.v); }

INLINE vec_f32x16 sc_sqrt(vec_f32x16 const &a) { return _mm512_sqrt_ps(a.v); }
// INLINE vec_f32x16 sc_rsqrt(vec_f32x16 const &a) { return _mm512_rsqrt_ps(a.v); }

INLINE vec_f32x16 sc_abs(vec_f32x16 const &a) {
    return _mm512_abs_ps(a.v);
}

inline __mmask16 sc_isnan(vec_f32x16 v1, vec_f32x16 v2) {
    return _mm512_cmp_ps_mask(v1.v, v2.v, _CMP_UNORD_Q);
}

inline __mmask16 sc_isnan(vec_f32x16 v1) {
    return _mm512_cmp_ps_mask(v1.v, v1.v, _CMP_UNORD_Q);
}

} // namespace kun_simd
#endif
#endif
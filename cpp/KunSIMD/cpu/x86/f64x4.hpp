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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F64X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F64X4_HPP
#include <immintrin.h>
#include <stdint.h>
#ifndef __AVX512F__
#include <cmath>
#endif
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {

template <>
struct alignas(32) vec<double, 4> {
  public:
    union {
        __m256d v;
        double raw[4];
    };

    using Masktype = vec<double, 4>;
    using T = double;
    static constexpr int lanes = 4;

    INLINE vec() = default;
    INLINE vec(double f) { v = _mm256_set1_pd(f); }
    INLINE vec(double i0, double i1, double i2, double i3) {
        v = _mm256_setr_pd(i0, i1, i2, i3);
    }
    INLINE vec(__m256d const &x) { v = x; }

    static INLINE vec load(const double *p) { return _mm256_loadu_pd(p); }
    static INLINE vec load_aligned(const double *p) { return _mm256_load_pd(p); }
    static INLINE void store(vec v, double *p) { _mm256_storeu_pd(p, v.v); }
    static INLINE void store_aligned(vec v, double *p) {
        _mm256_store_pd(p, v.v);
    }
    static INLINE vec masked_load(const double *p, Masktype mask) {
        return _mm256_maskload_pd(p, _mm256_castpd_si256(mask));
    }
    static INLINE void masked_store(vec v, double *p, Masktype mask) {
        _mm256_maskstore_pd(p, _mm256_castpd_si256(mask), v.v);
    }

    static Masktype make_mask(int N);
    operator __m256d() const { return v; }
};

using vec_f64x4 = vec<double, 4>;

INLINE vec_f64x4 operator+(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_add_pd(a.v, b.v);
}

INLINE vec_f64x4 operator-(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_sub_pd(a.v, b.v);
}

INLINE vec_f64x4 operator*(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_mul_pd(a.v, b.v);
}

INLINE vec_f64x4 operator/(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_div_pd(a.v, b.v);
}

INLINE vec_f64x4 sc_select(vec_f64x4 cond, vec_f64x4 const &a,
                           vec_f64x4 const &b) {
    return _mm256_blendv_pd(b, a, cond);
}

INLINE vec_f64x4 operator==(vec_f64x4 const &a, vec_f64x4 const &b) {
    auto ret = _mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ);
    return ret;
}
INLINE vec_f64x4 operator!=(vec_f64x4 const &a, vec_f64x4 const &b) {
    auto ret = _mm256_cmp_pd(a.v, b.v, _CMP_NEQ_OQ);
    return ret;
}
INLINE vec_f64x4 operator>(vec_f64x4 const &a, vec_f64x4 const &b) {
    auto ret = _mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ);
    return ret;
}
INLINE vec_f64x4 operator<(vec_f64x4 const &a, vec_f64x4 const &b) {
    auto ret = _mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ);
    return ret;
}
INLINE vec_f64x4 operator>=(vec_f64x4 const &a, vec_f64x4 const &b) {
    auto ret = _mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ);
    return ret;
}
INLINE vec_f64x4 operator<=(vec_f64x4 const &a, vec_f64x4 const &b) {
    auto ret = _mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ);
    return ret;
}

INLINE vec_f64x4 operator|(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_or_pd(a, b);
}
INLINE vec_f64x4 operator&(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_and_pd(a, b);
}
INLINE vec_f64x4 operator!(vec_f64x4 a) {
    return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
}

INLINE vec_f64x4 sc_fmadd(vec_f64x4 const &a, vec_f64x4 const &b,
                          vec_f64x4 const &c) {
#ifdef __FMA__
    return _mm256_fmadd_pd(a.v, b.v, c.v);
#else
    // Fallback to manual multiplication and addition if FMA is not available
    return _mm256_add_pd(_mm256_mul_pd(a.v, b.v), c.v);
#endif
}

INLINE vec_f64x4 sc_fmsub(vec_f64x4 const &a, vec_f64x4 const &b,
                          vec_f64x4 const &c) {
#ifdef __FMA__
    return _mm256_fmsub_pd(a.v, b.v, c.v);
#else
    // Fallback to manual multiplication and subtraction if FMA is not available
    return _mm256_sub_pd(_mm256_mul_pd(a.v, b.v), c.v);
#endif
}

INLINE vec_f64x4 sc_fnmadd(vec_f64x4 const &a, vec_f64x4 const &b,
                           vec_f64x4 const &c) {
#ifdef __FMA__
    return _mm256_fnmadd_pd(a.v, b.v, c.v);
#else
    // Fallback to manual multiplication and negation if FMA is not available
    return _mm256_sub_pd(c.v, _mm256_mul_pd(a.v, b.v));
#endif
}

INLINE vec_f64x4 sc_max(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_max_pd(a.v, b.v);
}
INLINE vec_f64x4 sc_min(vec_f64x4 const &a, vec_f64x4 const &b) {
    return _mm256_min_pd(a.v, b.v);
}

INLINE vec_f64x4 sc_round(vec_f64x4 const &a) {
    return _mm256_round_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f64x4 sc_ceil(vec_f64x4 const &a) { return _mm256_ceil_pd(a.v); }
INLINE vec_f64x4 sc_floor(vec_f64x4 const &a) { return _mm256_floor_pd(a.v); }

INLINE vec_f64x4 sc_sqrt(vec_f64x4 const &a) { return _mm256_sqrt_pd(a.v); }

INLINE vec_f64x4 sc_abs(vec_f64x4 const &a) {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0f), a.v);
}

inline vec_f64x4 sc_isnan(vec_f64x4 v1, vec_f64x4 v2) {
    return _mm256_cmp_pd(v1.v, v2.v, _CMP_UNORD_Q);
}

inline vec_f64x4 sc_isnan(vec_f64x4 v1) {
    return _mm256_cmp_pd(v1.v, v1.v, _CMP_UNORD_Q);
}

} // namespace kun_simd
#endif
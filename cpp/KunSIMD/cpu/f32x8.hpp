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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_F32X8_HPP
#include <immintrin.h>
#include <stdint.h>
#ifndef __AVX512F__
#include <cmath>
#endif
#include "common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd{

template<>
struct vec<float, 8> {
public:
    union {
        __m256 v;
        float raw[8];
    } __attribute__((aligned(32)));

    INLINE vec() = default;
    INLINE vec(float f) { v = _mm256_set1_ps(f); }
    INLINE vec(float i0, float i1, float i2, float i3, float i4, float i5,
            float i6, float i7) {
        v = _mm256_setr_ps(i0, i1, i2, i3, i4, i5, i6, i7);
    }
    INLINE vec(__m256 const &x) { v = x; }

    static INLINE vec load(const float *p) { return _mm256_loadu_ps(p); }
    static INLINE vec load_aligned(const float *p) {
        return _mm256_load_ps(p);
    }
    static INLINE void store(vec v, float *p) {
        _mm256_storeu_ps(p, v.v);
    }
    static INLINE void store_aligned(vec v, float *p) {
        _mm256_store_ps(p, v.v);
    }

    operator __m256() const {
        return v;
    }
};

using vec_f32x8 = vec<float, 8>;

INLINE vec_f32x8 operator+(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_add_ps(a.v, b.v);
}

INLINE vec_f32x8 operator-(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_sub_ps(a.v, b.v);
}

INLINE vec_f32x8 operator*(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_mul_ps(a.v, b.v);
}

INLINE vec_f32x8 operator/(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_div_ps(a.v, b.v);
}

INLINE vec_f32x8 sc_select(
        vec_f32x8 cond, vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_blendv_ps(b, a, cond);
}

INLINE vec_f32x8 operator==(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ);
    return ret;
}
INLINE vec_f32x8 operator!=(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_NEQ_OQ);
    return ret;
}
INLINE vec_f32x8 operator>(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ);
    return ret;
}
INLINE vec_f32x8 operator<(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ);
    return ret;
}
INLINE vec_f32x8 operator>=(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ);
    return ret;
}
INLINE vec_f32x8 operator<=(vec_f32x8 const &a, vec_f32x8 const &b) {
    auto ret = _mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ);
    return ret;
}

INLINE vec_f32x8 sc_fmadd(
        vec_f32x8 const &a, vec_f32x8 const &b, vec_f32x8 const &c) {
    return _mm256_fmadd_ps(a.v, b.v, c.v);
}

INLINE vec_f32x8 sc_max(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_max_ps(a.v, b.v);
}
INLINE vec_f32x8 sc_min(vec_f32x8 const &a, vec_f32x8 const &b) {
    return _mm256_min_ps(a.v, b.v);
}

INLINE vec_f32x8 sc_round(vec_f32x8 const &a) {
    return _mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

INLINE vec_f32x8 sc_ceil(vec_f32x8 const &a) {
    return _mm256_ceil_ps(a.v);
}
INLINE vec_f32x8 sc_floor(vec_f32x8 const &a) {
    return _mm256_floor_ps(a.v);
}

INLINE vec_f32x8 sc_sqrt(vec_f32x8 const &a) {
    return _mm256_sqrt_ps(a.v);
}
INLINE vec_f32x8 sc_rsqrt(vec_f32x8 const &a) {
    return _mm256_rsqrt_ps(a.v);
}

INLINE vec_f32x8 sc_abs(vec_f32x8 const &a) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.v);
}

inline vec_f32x8 sc_isnan(vec_f32x8 v1, vec_f32x8 v2) {
    return _mm256_cmp_ps(v1.v, v2.v, _CMP_UNORD_Q);
}

inline vec_f32x8 sc_isnan(vec_f32x8 v1) {
    return _mm256_cmp_ps(v1.v, v1.v, _CMP_UNORD_Q);
}

}
#endif
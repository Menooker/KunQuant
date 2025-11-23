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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S32X8_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S32X8_HPP
#include <immintrin.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/cpu/x86/common.hpp>
#include <KunSIMD/Vector.hpp>

namespace kun_simd {
template<>
struct alignas(32) vec<int32_t, 8> {
public:
    union {
        __m256i v;
        int32_t raw[8];
    };
    using Masktype = vec<int32_t, 8>;
    using T = int32_t;
    static constexpr int lanes = 8;

    INLINE vec() = default;
    INLINE vec(int32_t f) { v = _mm256_set1_epi32(f); }
    INLINE vec(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4,
            int32_t i5, int32_t i6, int32_t i7) {
        v = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
    }
    INLINE vec(__m256i const &x) { v = x; }
    // INLINE operator vec_f32x8() const;

    static INLINE vec load(const int32_t *p) {
        return _mm256_loadu_si256((const __m256i *)p);
    }
    static INLINE vec load_aligned(const int32_t *p) {
        return _mm256_load_si256((const __m256i *)p);
    }
    static INLINE void store(vec v, int32_t *p) {
        _mm256_storeu_si256((__m256i *)p, v.v);
    }
    static INLINE void store_aligned(vec v, int32_t *p) {
        _mm256_store_si256((__m256i *)p, v.v);
    }
    operator __m256i() const { return v; }
};

using vec_s32x8 = vec<int32_t, 8>;

INLINE vec_s32x8 operator+(vec_s32x8 const &a, vec_s32x8 const &b) {
    AVX_IMPL(_mm_add_epi32, _mm256_add_epi32, a.v, b.v);
}

INLINE vec_s32x8 operator-(vec_s32x8 const &a, vec_s32x8 const &b) {
    AVX_IMPL(_mm_sub_epi32, _mm256_sub_epi32, a.v, b.v);
}


#ifdef __AVX2__
INLINE vec_s32x8 operator-(vec_s32x8 const &a) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), a.v);
}

INLINE vec_s32x8 operator*(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_mullo_epi32(a.v, b.v);
}
#endif

// INLINE vec_s32x8 operator/(vec_s32x8 const &a, vec_s32x8 const &b) {
//     return _mm256_div_epi32(a.v, b.v);
// }
#ifdef __AVX2__
INLINE vec_s32x8 operator~(vec_s32x8 const &a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
}
#endif
INLINE vec_s32x8 operator&(vec_s32x8 const &a, vec_s32x8 const &b) {
    AVX_USE_FP_OP(and, ps, a.v, b.v);
}
INLINE vec_s32x8 operator|(vec_s32x8 const &a, vec_s32x8 const &b) {
    AVX_USE_FP_OP(or, ps, a.v, b.v);
}
INLINE vec_s32x8 operator^(vec_s32x8 const &a, vec_s32x8 const &b) {
    AVX_USE_FP_OP(xor, ps, a.v, b.v);
}

#if 0
INLINE __mmask8 operator!(vec_s32x8 const &a) {
    return _mm256_cmp_epi32_mask(a.v, _mm256_setzero_si256(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_s32x8 sc_select(
        __mmask8 mask, vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_mask_blend_epi32(mask, b.v, a.v);
}
#else
INLINE vec_s32x8 operator==(vec_s32x8 const &a, vec_s32x8 const &b) {
    AVX_IMPL(_mm_cmpeq_epi32, _mm256_cmpeq_epi32, a.v, b.v);
}
INLINE vec_s32x8 operator>(vec_s32x8 const &a, vec_s32x8 const &b) {
    AVX_IMPL(_mm_cmpgt_epi32, _mm256_cmpgt_epi32, a.v, b.v);
}
#endif

namespace {
INLINE vec_s32x8 blendvps_si256(__m256i a, __m256i b, __m256i mask) {
    __m256 res =
        _mm256_blendv_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b),
                         _mm256_castsi256_ps(mask));
    return _mm256_castps_si256(res);
}
}

INLINE vec_s32x8 sc_select(
        vec_s32x8 mask, vec_s32x8 const &a, vec_s32x8 const &b) {
    return blendvps_si256(b, a, mask);
}

#ifdef __AVX2__
INLINE vec_s32x8 operator<<(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_sllv_epi32(a.v, b.v);
}
INLINE vec_s32x8 operator>>(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_srav_epi32(a.v, b.v);
}

INLINE vec_s32x8 logical_shr(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_srlv_epi32(a.v, b.v);
}
#endif

template <int v>
INLINE vec_s32x8 logical_shl(vec_s32x8 const &a) {
    AVX_CONST_IMPL(_mm_slli_epi32, _mm256_slli_epi32, a.v, v);
}

template <int v>
INLINE vec_s32x8 logical_shr(vec_s32x8 const &a) {
    AVX_CONST_IMPL(_mm_srli_epi32, _mm256_srli_epi32, a.v, v);
}

// operator /

INLINE vec_s32x8 sc_max(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_max_epi32(a.v, b.v);
}
INLINE vec_s32x8 sc_min(vec_s32x8 const &a, vec_s32x8 const &b) {
    return _mm256_min_epi32(a.v, b.v);
}

INLINE vec_s32x8 sc_abs(vec_s32x8 const &a) {
    return _mm256_abs_epi32(a.v);
}
}

#endif
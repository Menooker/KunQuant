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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S64X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_S64X4_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {
template<>
struct alignas(32) vec<int64_t, 4> {
public:
    union {
        __m256i v;
        int64_t raw[4];
    };

    using T = int64_t;
    static constexpr int lanes = 4;

    INLINE vec() = default;
    INLINE vec(int64_t f) { v = _mm256_set1_epi64x(f); }
    INLINE vec(int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        v = _mm256_setr_epi64x(i0, i1, i2, i3);
    }
    INLINE vec(__m256i const &x) { v = x; }
    // INLINE operator vec_f32x8() const;

    static INLINE vec load(const int64_t *p) {
        return _mm256_loadu_si256((const __m256i *)p);
    }
    static INLINE vec load_aligned(const int64_t *p) {
        return _mm256_load_si256((const __m256i *)p);
    }
    static INLINE void store(vec v, int64_t *p) {
        _mm256_storeu_si256((__m256i *)p, v.v);
    }
    static INLINE void store_aligned(vec v, int64_t *p) {
        _mm256_store_si256((__m256i *)p, v.v);
    }
    operator __m256i() const { return v; }
};

using vec_s64x4 = vec<int64_t, 4>;

INLINE vec_s64x4 operator+(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_add_epi64(a.v, b.v);
}

INLINE vec_s64x4 operator-(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_sub_epi64(a.v, b.v);
}
INLINE vec_s64x4 operator-(vec_s64x4 const &a) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), a.v);
}

INLINE vec_s64x4 operator*(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_mullo_epi64(a.v, b.v);
}

// INLINE vec_s64x4 operator/(vec_s64x4 const &a, vec_s64x4 const &b) {
//     return _mm256_div_epi64(a.v, b.v);
// }

INLINE vec_s64x4 operator~(vec_s64x4 const &a) {
    return _mm256_xor_si256(a.v, _mm256_set1_epi64x(-1));
}
INLINE vec_s64x4 operator&(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_and_si256(a.v, b.v);
}
INLINE vec_s64x4 operator|(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_or_si256(a.v, b.v);
}
INLINE vec_s64x4 operator^(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_xor_si256(a.v, b.v);
}


INLINE vec_s64x4 operator==(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_cmpeq_epi64(a, b);
}

namespace {
INLINE vec_s64x4 blendvpd_si256(__m256i a, __m256i b, __m256i mask) {
    __m256d res =
        _mm256_blendv_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b),
                         _mm256_castsi256_pd(mask));
    return _mm256_castpd_si256(res);
}
}

INLINE vec_s64x4 sc_select(
        vec_s64x4 mask, vec_s64x4 const &a, vec_s64x4 const &b) {
    return blendvpd_si256(b, a, mask);
}

INLINE vec_s64x4 operator<<(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_sllv_epi64(a.v, b.v);
}
INLINE vec_s64x4 operator>>(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_srav_epi64(a.v, b.v);
}

// operator /

INLINE vec_s64x4 sc_max(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_max_epi64(a.v, b.v);
}
INLINE vec_s64x4 sc_min(vec_s64x4 const &a, vec_s64x4 const &b) {
    return _mm256_min_epi64(a.v, b.v);
}

INLINE vec_s64x4 sc_abs(vec_s64x4 const &a) {
    return _mm256_abs_epi64(a.v);
}
}

#endif
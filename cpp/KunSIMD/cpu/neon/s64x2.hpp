/*******************************************************************************
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_S64X2_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_S64X2_HPP

#include <arm_neon.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {

template<>
struct alignas(16) vec<int64_t, 2> {
public:
    union {
        int64x2_t v;
        int64_t raw[2];
    };
    using Masktype = vec<int64_t, 2>;
    using T = int64_t;
    static constexpr int lanes = 2;

    INLINE vec() = default;
    INLINE vec(int64_t f) { v = vdupq_n_s64(f); }
    INLINE vec(int64_t i0, int64_t i1) { v = int64x2_t {i0, i1}; }
    INLINE vec(int64x2_t const &x) { v = x; }
    INLINE vec(uint64x2_t const &x) { v = vreinterpretq_s64_u64(x); }

    static INLINE vec load(const int64_t *p) { return vld1q_s64(p); }
    static INLINE vec load_aligned(const int64_t *p) { return vld1q_s64(p); }
    static INLINE void store(vec v, int64_t *p) { vst1q_s64(p, v.v); }
    static INLINE void store_aligned(vec v, int64_t *p) { vst1q_s64(p, v.v); }

    operator int64x2_t() const { return v; }
};

using vec_s64x2 = vec<int64_t, 2>;

INLINE vec_s64x2 operator+(vec_s64x2 const &a, vec_s64x2 const &b) {
    return vaddq_s64(a.v, b.v);
}

INLINE vec_s64x2 operator-(vec_s64x2 const &a, vec_s64x2 const &b) {
    return vsubq_s64(a.v, b.v);
}
INLINE vec_s64x2 operator-(vec_s64x2 const &a) {
    return vnegq_s64(a.v);
}

INLINE vec_s64x2 operator*(vec_s64x2 const &a, vec_s64x2 const &b) {
    // 64-bit integer vector multiply isn't universally available as a single NEON intrinsic,
    // do element-wise scalar multiply to be portable.
    return {a.raw[0] * b.raw[0], a.raw[1] * b.raw[1]};
}

INLINE vec_s64x2 operator!(vec_s64x2 const &a) {
    return vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(a.v)));
}
INLINE vec_s64x2 operator~(vec_s64x2 const &a) {
    return !a;
}
INLINE vec_s64x2 operator&(vec_s64x2 const &a, vec_s64x2 const &b) {
    return vandq_s64(a.v, b.v);
}
INLINE vec_s64x2 operator|(vec_s64x2 const &a, vec_s64x2 const &b) {
    return vorrq_s64(a.v, b.v);
}
INLINE vec_s64x2 operator^(vec_s64x2 const &a, vec_s64x2 const &b) {
    return veorq_s64(a.v, b.v);
}

INLINE vec_s64x2 operator==(vec_s64x2 const &a, vec_s64x2 const &b) {
    return vceqq_s64(a.v, b.v);
}
INLINE vec_s64x2 operator>(vec_s64x2 const &a, vec_s64x2 const &b) {
    return vcgtq_s64(a.v, b.v);
}

INLINE vec_s64x2 sc_select(vec_s64x2 mask, vec_s64x2 const &a, vec_s64x2 const &b) {
    uint64x2_t m = vreinterpretq_u64_s64(mask.v);
    return vbslq_s64(m, a.v, b.v);
}

INLINE vec_s64x2 operator<<(vec_s64x2 const &a, vec_s64x2 const &b) {
    return vshlq_s64(a.v, b.v);
}
INLINE vec_s64x2 operator>>(vec_s64x2 const &a, vec_s64x2 const &b) {
    int64x2_t neg = vnegq_s64(b.v);
    return vshlq_s64(a.v, neg);
}


INLINE vec_s64x2 logical_shr(vec_s64x2 const &a, vec_s64x2 const &b) {
    uint64x2_t ua = vreinterpretq_u64_s64(a.v);
    int64x2_t neg = vnegq_s64(b.v);
    uint64x2_t r = vshlq_u64(ua, neg);
    return vreinterpretq_s64_u64(r);
}

template <int v>
INLINE vec_s64x2 logical_shl(vec_s64x2 const &a) {
    return a << vec_s64x2{v};
}

template <int v>
INLINE vec_s64x2 logical_shr(vec_s64x2 const &a) {
    return logical_shr(a, vec_s64x2(v));
}

} // namespace

#endif
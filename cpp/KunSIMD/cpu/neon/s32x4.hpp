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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_S32X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_S32X4_HPP

#include <arm_neon.h>
#include <stdint.h>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>

namespace kun_simd {

template<>
struct alignas(16) vec<int32_t, 4> {
public:
    union {
        int32x4_t v;
        int32_t raw[4];
    };
    using Masktype = vec<int32_t, 4>;
    using T = int32_t;
    static constexpr int lanes = 4;

    INLINE vec() = default;
    INLINE vec(int32_t f) { v = vdupq_n_s32(f); }
    INLINE vec(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
        v = int32x4_t {i0, i1, i2, i3};
    }
    INLINE vec(int32x4_t const &x) { v = x; }
    INLINE vec(uint32x4_t const &x) { v = vreinterpretq_s32_u32(x); }

    static INLINE vec load(const int32_t *p) { return vld1q_s32(p); }
    static INLINE vec load_aligned(const int32_t *p) { return vld1q_s32(p); }
    static INLINE void store(vec v, int32_t *p) { vst1q_s32(p, v.v); }
    static INLINE void store_aligned(vec v, int32_t *p) { vst1q_s32(p, v.v); }

    operator int32x4_t() const { return v; }
};

using vec_s32x4 = vec<int32_t, 4>;

INLINE vec_s32x4 operator+(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vaddq_s32(a.v, b.v);
}

INLINE vec_s32x4 operator-(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vsubq_s32(a.v, b.v);
}
INLINE vec_s32x4 operator-(vec_s32x4 const &a) {
    return vnegq_s32(a.v);
}

INLINE vec_s32x4 operator*(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vmulq_s32(a.v, b.v);
}

INLINE vec_s32x4 operator!(vec_s32x4 const &a) {
    return vmvnq_s32(a.v);
}
INLINE vec_s32x4 operator~(vec_s32x4 const &a) {
    return !a;
}
INLINE vec_s32x4 operator&(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vandq_s32(a.v, b.v);
}
INLINE vec_s32x4 operator|(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vorrq_s32(a.v, b.v);
}
INLINE vec_s32x4 operator^(vec_s32x4 const &a, vec_s32x4 const &b) {
    return veorq_s32(a.v, b.v);
}

INLINE vec_s32x4 operator==(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vceqq_s32(a.v, b.v);
}
INLINE vec_s32x4 operator>(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vcgtq_s32(a.v, b.v);
}

INLINE vec_s32x4 sc_select(vec_s32x4 mask, vec_s32x4 const &a, vec_s32x4 const &b) {
    uint32x4_t m = vreinterpretq_u32_s32(mask.v);
    return vbslq_s32(m, a.v, b.v);
}

INLINE vec_s32x4 operator<<(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vshlq_s32(a.v, b.v);
}
INLINE vec_s32x4 operator>>(vec_s32x4 const &a, vec_s32x4 const &b) {
    int32x4_t neg = vnegq_s32(b.v);
    return vshlq_s32(a.v, neg);
}


INLINE vec_s32x4 logical_shr(vec_s32x4 const &a, vec_s32x4 const &b) {
    uint32x4_t ua = vreinterpretq_u32_s32(a.v);
    int32x4_t neg = vnegq_s32(b.v);
    uint32x4_t r = vshlq_u32(ua, neg);
    return vreinterpretq_s32_u32(r);
}

INLINE vec_s32x4 sc_max(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vmaxq_s32(a.v, b.v);
}
INLINE vec_s32x4 sc_min(vec_s32x4 const &a, vec_s32x4 const &b) {
    return vminq_s32(a.v, b.v);
}

INLINE vec_s32x4 sc_abs(vec_s32x4 const &a) {
    return vabsq_s32(a.v);
}

template <int v>
INLINE vec_s32x4 logical_shl(vec_s32x4 const &a) {
    return a << vec_s32x4{v};
}

template <int v>
INLINE vec_s32x4 logical_shr(vec_s32x4 const &a) {
    return logical_shr(a, vec_s32x4(v));
}

} // namespace kun_simd

#endif

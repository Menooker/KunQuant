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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_F32X4_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_F32X4_HPP

#include <arm_neon.h>
#include <stdint.h>
#include <cmath>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>
#include "s32x4.hpp"

namespace kun_simd {

template <>
struct alignas(16) vec<float, 4> {
  public:
    union {
        float32x4_t v;
        float raw[4];
    };
    using Masktype = vec<int32_t, 4>;
    using T = float;
    static constexpr int lanes = 4;

    INLINE vec() = default;
    INLINE vec(float f) { v = vdupq_n_f32(f); }
    INLINE vec(float i0, float i1, float i2, float i3) {
        v = float32x4_t{i0, i1, i2, i3};
    }
    INLINE vec(float32x4_t const &x) { v = x; }

    static INLINE vec load(const float *p) { return vld1q_f32(p); }
    static INLINE vec load_aligned(const float *p) { return vld1q_f32(p); }
    static INLINE void store(vec v, float *p) { vst1q_f32(p, v.v); }
    static INLINE void store_aligned(vec v, float *p) { vst1q_f32(p, v.v); }

    operator float32x4_t() const { return v; }
};

using vec_f32x4 = vec<float, 4>;
using vec_s32x4 = vec<int32_t, 4>;

INLINE vec_f32x4 operator+(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vaddq_f32(a.v, b.v);
}

INLINE vec_f32x4 operator-(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vsubq_f32(a.v, b.v);
}

INLINE vec_f32x4 operator*(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vmulq_f32(a.v, b.v);
}

INLINE vec_f32x4 operator/(vec_f32x4 const &a, vec_f32x4 const &b) {
    float out[4];
    for (int i = 0; i < 4; ++i) out[i] = a.raw[i] / b.raw[i];
    return vld1q_f32(out);
}

/* selection using an integer mask (vec<int32_t,4>) */
INLINE vec_f32x4 sc_select(vec_s32x4 const &cond, vec_f32x4 const &a,
                           vec_f32x4 const &b) {
    // vbslq_f32 expects a uint32x4_t mask
    uint32x4_t mask_u = vreinterpretq_u32_s32(cond.v);
    return vbslq_f32(mask_u, a.v, b.v);
}

/* comparison -> return integer mask (vec<int32_t,4>) */
INLINE vec_s32x4 operator==(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vreinterpretq_s32_u32(vceqq_f32(a.v, b.v));
}
INLINE vec_s32x4 operator!=(vec_f32x4 const &a, vec_f32x4 const &b) {
    uint32x4_t cmp = vceqq_f32(a.v, b.v);
    return vreinterpretq_s32_u32(vmvnq_u32(cmp));
}
INLINE vec_s32x4 operator>(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vreinterpretq_s32_u32(vcgtq_f32(a.v, b.v));
}
INLINE vec_s32x4 operator<(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vreinterpretq_s32_u32(vcltq_f32(a.v, b.v));
}
INLINE vec_s32x4 operator>=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vreinterpretq_s32_u32(vcgeq_f32(a.v, b.v));
}
INLINE vec_s32x4 operator<=(vec_f32x4 const &a, vec_f32x4 const &b) {
    return vreinterpretq_s32_u32(vcleq_f32(a.v, b.v));
}

/* keep float bitwise ops (reinterpreted) for completeness */
INLINE vec_f32x4 sc_fmadd(vec_f32x4 const &a, vec_f32x4 const &b,
                          vec_f32x4 const &c) {
    return vfmaq_f32(c.v, a.v, b.v);
}

INLINE vec_f32x4 sc_fmsub(vec_f32x4 const &a, vec_f32x4 const &b,
                          vec_f32x4 const &c) {
    // a*b-c                
    return vfmaq_f32(vnegq_f32(c.v), a.v, b.v);
}

INLINE vec_f32x4 sc_fnmadd(vec_f32x4 const &a, vec_f32x4 const &b,
                           vec_f32x4 const &c) {
    // c-a*b
    return vfmsq_f32(c.v, a.v, b.v);
}

INLINE vec_f32x4 sc_max(vec_f32x4 const &a, vec_f32x4 const &b) {
    // vminnmq_f32 has different behavior when NAN is involved
    return sc_select(a > b, a, b);
}
INLINE vec_f32x4 sc_min(vec_f32x4 const &a, vec_f32x4 const &b) {
    // vminnmq_f32 has different behavior when NAN is involved
    return sc_select(a < b, a, b);
}

INLINE vec_f32x4 sc_round(vec_f32x4 const &a) {
    // round to nearest using NEON: convert to int with rounding-to-nearest then back
    int32x4_t t = vcvtnq_s32_f32(a.v);
    return vcvtq_f32_s32(t);
}

INLINE vec_f32x4 sc_ceil(vec_f32x4 const &a) {
    return vrndpq_f32(a.v);
}

INLINE vec_f32x4 sc_floor(vec_f32x4 const &a) {
    return vrndmq_f32(a.v);
}

INLINE vec_f32x4 sc_sqrt(vec_f32x4 const &a) {
    return vsqrtq_f32(a.v);
}

INLINE vec_f32x4 sc_abs(vec_f32x4 const &a) {
    return vabsq_f32(a.v);
}


INLINE vec_s32x4 sc_isnan(vec_f32x4 v1) {
    vec_s32x4 ret;
    uint32x4_t cmp = vceqq_f32(v1.v, v1.v);
    uint32x4_t nan_u = vmvnq_u32(cmp);
    return vreinterpretq_s32_u32(nan_u);
}

INLINE vec_s32x4 sc_isnan(vec_f32x4 v1, vec_f32x4 v2) {
    return sc_isnan(v1) | sc_isnan(v2);
}

} // namespace kun_simd
#endif
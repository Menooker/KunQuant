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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_F64X2_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_KERNEL_INCLUDE_ARM_NEON_VEC_F64X2_HPP

#include <arm_neon.h>
#include <stdint.h>
#include <cmath>
#include "../common.hpp"
#include <KunSIMD/Vector.hpp>
#include "s64x2.hpp"

namespace kun_simd {

template <>
struct alignas(16) vec<double, 2> {
  public:
    union {
        float64x2_t v;
        double raw[2];
    };
    using Masktype = vec<int64_t, 2>;
    using T = double;
    static constexpr int lanes = 2;

    INLINE vec() = default;
    INLINE vec(double f) { v = vdupq_n_f64(f); }
    INLINE vec(double i0, double i1) { v = float64x2_t{i0, i1}; }
    INLINE vec(float64x2_t const &x) { v = x; }

    static INLINE vec load(const double *p) { return vld1q_f64(p); }
    static INLINE vec load_aligned(const double *p) { return vld1q_f64(p); }
    static INLINE void store(vec v, double *p) { vst1q_f64(p, v.v); }
    static INLINE void store_aligned(vec v, double *p) { vst1q_f64(p, v.v); }

    operator float64x2_t() const { return v; }
};

using vec_f64x2 = vec<double, 2>;
using vec_s64x2 = vec<int64_t, 2>;

INLINE vec_f64x2 operator+(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vaddq_f64(a.v, b.v);
}

INLINE vec_f64x2 operator-(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vsubq_f64(a.v, b.v);
}

INLINE vec_f64x2 operator*(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vmulq_f64(a.v, b.v);
}

INLINE vec_f64x2 operator/(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vec_f64x2{a.raw[0] / b.raw[0], a.raw[1] / b.raw[1]};
}

/* selection using an integer mask (vec<int64_t,2>) */
INLINE vec_f64x2 sc_select(vec_s64x2 const &cond, vec_f64x2 const &a,
                           vec_f64x2 const &b) {
    // vbslq_f64 expects a uint64x2_t mask
    uint64x2_t mask_u = vreinterpretq_u64_s64(cond.v);
    return vbslq_f64(mask_u, a.v, b.v);
}

/* comparison -> return integer mask (vec<int64_t,2>) */
INLINE vec_s64x2 operator==(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vreinterpretq_s64_u64(vceqq_f64(a.v, b.v));
}
INLINE vec_s64x2 operator!=(vec_f64x2 const &a, vec_f64x2 const &b) {
    uint64x2_t cmp = vceqq_f64(a.v, b.v);
    return !vec_s64x2{cmp};
}
INLINE vec_s64x2 operator>(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vreinterpretq_s64_u64(vcgtq_f64(a.v, b.v));
}
INLINE vec_s64x2 operator<(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vreinterpretq_s64_u64(vcltq_f64(a.v, b.v));
}
INLINE vec_s64x2 operator>=(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vreinterpretq_s64_u64(vcgeq_f64(a.v, b.v));
}
INLINE vec_s64x2 operator<=(vec_f64x2 const &a, vec_f64x2 const &b) {
    return vreinterpretq_s64_u64(vcleq_f64(a.v, b.v));
}

/* keep float bitwise ops (reinterpreted) for completeness */
INLINE vec_f64x2 sc_fmadd(vec_f64x2 const &a, vec_f64x2 const &b,
                          vec_f64x2 const &c) {
    // a*b + c
    return vfmaq_f64(c.v, a.v, b.v);
}

INLINE vec_f64x2 sc_fmsub(vec_f64x2 const &a, vec_f64x2 const &b,
                          vec_f64x2 const &c) {
    // a*b-c                
    return vfmaq_f64(vnegq_f64(c.v), a.v, b.v);
}

INLINE vec_f64x2 sc_fnmadd(vec_f64x2 const &a, vec_f64x2 const &b,
                           vec_f64x2 const &c) {
    // c-a*b
    return vfmsq_f64(c.v, a.v, b.v);
}

INLINE vec_f64x2 sc_max(vec_f64x2 const &a, vec_f64x2 const &b) {
    return sc_select(a > b, a, b);
}
INLINE vec_f64x2 sc_min(vec_f64x2 const &a, vec_f64x2 const &b) {
    return sc_select(a < b, a, b);
}

INLINE vec_f64x2 sc_round(vec_f64x2 const &a) {
    // round to nearest using NEON: convert to int64 with rounding-to-nearest then back
    int64x2_t t = vcvtnq_s64_f64(a.v);
    return vcvtq_f64_s64(t);
}

INLINE vec_f64x2 sc_ceil(vec_f64x2 const &a) {
    return vrndpq_f64(a.v);
}

INLINE vec_f64x2 sc_floor(vec_f64x2 const &a) {
    return vrndmq_f64(a.v);
}

INLINE vec_f64x2 sc_sqrt(vec_f64x2 const &a) {
    return vsqrtq_f64(a.v);
}

INLINE vec_f64x2 sc_abs(vec_f64x2 const &a) {
    return vabsq_f64(a.v);
}

INLINE vec_s64x2 sc_isnan(vec_f64x2 v1) {
    uint64x2_t cmp = vceqq_f64(v1.v, v1.v);
    return !vec_s64x2{cmp};
}

INLINE vec_s64x2 sc_isnan(vec_f64x2 v1, vec_f64x2 v2) {
    return sc_isnan(v1) | sc_isnan(v2);
}

} // namespace kun_simd
#endif
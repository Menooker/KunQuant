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

#pragma once

#include "f32x4.hpp"
#include "s32x4.hpp"
#include "f64x2.hpp"
#include "s64x2.hpp"

namespace kun_simd {

template<int scale>
INLINE vec_f32x4 gather(const float* ptr, vec_s32x4 v) {
    float out[4];
    for (int i = 0; i < 4; ++i) {
        out[i] = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(ptr) + v.raw[i] * scale);
    }
    return vld1q_f32(out);
}

template<int scale>
INLINE vec_f64x2 gather(const double* ptr, vec_s64x2 v) {
    double out[2];
    for (int i = 0; i < 2; ++i) {
        out[i] = *reinterpret_cast<const double*>(reinterpret_cast<const char*>(ptr) + v.raw[i] * scale);
    }
    return vld1q_f64(out);
}

}
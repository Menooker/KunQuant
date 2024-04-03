#pragma once

#include "Base.hpp"
#include <cmath>
#include <limits>
#include <stdint.h>
#include <KunSIMD/cpu/Math.hpp>

namespace kun {
namespace ops {

inline kun_simd::vec_f32x8 LogFast(kun_simd::vec_f32x8 v) {
    return kun_simd::log<float, 8>(v);
}

inline kun_simd::vec_f32x8 Exp(kun_simd::vec_f32x8 v) {
    return kun_simd::exp<float, 8>(v);
}
}
}
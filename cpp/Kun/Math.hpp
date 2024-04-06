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

template <typename T, int lanes>
inline kun_simd::vec<T, lanes> Exp(kun_simd::vec<T, lanes> v) {
    return kun_simd::exp(v);
}

}
}
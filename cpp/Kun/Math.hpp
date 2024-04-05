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


inline kun_simd::vec<double, 4> Exp(kun_simd::vec<double, 4> a) {
    double *v = a.raw;
    for (int i = 0; i < 4; i++) {
        v[i] = std::exp(v[i]);
    }
    return a;
}

template <typename T, int lanes>
inline kun_simd::vec<T, lanes> Exp(kun_simd::vec<T, lanes> v) {
    return kun_simd::exp(v);
}

}
}
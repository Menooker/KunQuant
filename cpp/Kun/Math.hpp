#pragma once

#include "Base.hpp"
#include <cmath>
#include <limits>
#include <stdint.h>
#include <KunSIMD/cpu/Math.hpp>

namespace kun {
namespace ops {

inline f32x8 Log(f32x8 v) {
    return kun_simd::log<float, 8>(v);
}

}
}
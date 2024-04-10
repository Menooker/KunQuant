#pragma once

#include "Base.hpp"
#include <cmath>
#include <limits>
#include <stdint.h>
#include <KunSIMD/cpu/Math.hpp>

namespace kun {
namespace ops {
template <typename T1>
struct DecayVec {
    using result = decltype(kun_simd::sc_abs(std::declval<T1>()));
};

template <typename T1>
using DecayVec_t = typename DecayVec<T1>::result;

template <typename T>
inline DecayVec_t<T> LogFast(T v) {
    return kun_simd::log(DecayVec_t<T>{v});
}

template <typename T>
inline DecayVec_t<T> Exp(T v) {
    return kun_simd::exp(DecayVec_t<T>{v});
}

}
}
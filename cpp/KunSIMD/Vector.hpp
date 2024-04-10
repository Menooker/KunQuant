#pragma once
#include <stdint.h>

namespace kun_simd {
template <typename T, int lanes>
struct vec {};


template <typename T>
struct fp_trait {

};


template <>
struct fp_trait<float> {
    using int_t = int32_t;
    static constexpr int exponent_bits = 8;
    static constexpr int fraction_bits = 23;
};

template <>
struct fp_trait<double> {
    using int_t = int64_t;
    static constexpr int exponent_bits = 11;
    static constexpr int fraction_bits = 52;
};

}
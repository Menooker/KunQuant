#pragma once
#include <stdint.h>

namespace kun_simd {
template <typename T, int lanes>
struct vec {};


template <typename T>
struct int_type_of_same_size {

};


template <>
struct int_type_of_same_size<float> {
    using result = int32_t;
};

template <>
struct int_type_of_same_size<double> {
    using result = int64_t;
};

}
#pragma once
#include <Kun/Base.hpp>

namespace kun_simd {

template <typename T>
struct LogLookupTable {
    KUN_API static const T r_table[32];
    KUN_API static const T logr_table[32];
};
}
#pragma once
#include <Kun/Base.hpp>

namespace kun_simd {

template <typename T>
struct LogLookupTable {
    KUN_API static const T r_table[32];
    KUN_API static const T logr_table[32];
};

// make clang happy about template var
#ifndef _MSC_VER
template <> const double LogLookupTable<double>::r_table[32];
template <> const double LogLookupTable<double>::logr_table[32];
template <> const float LogLookupTable<float>::r_table[32];
template <> const float LogLookupTable<float>::logr_table[32];
#endif
}
#pragma once

namespace kun_simd {

template <typename T1, typename T2>
inline T1 bitcast(T2 v) {
    static_assert(sizeof(T1) == sizeof(T2), "unmatched bitcast");
    union {
        T1 v1;
        T2 v2;
    } val;
    val.v2 = v;
    return val.v1;
}

} // namespace kun_simd

#if defined(__AVX__)
#include "x86/cast.hpp"
#else
#include "neon/cast.hpp"
#endif

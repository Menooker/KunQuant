#pragma once

#include <immintrin.h>

namespace kun {
    struct Context;
    using f32x8 = __m256;
}

#ifdef _MSC_VER
#define KUN_EXPORT __declspec(dllexport)
#ifdef KUN_CORE_LIB
#define KUN_API KUN_EXPORT
#else
#define KUN_API __declspec(dllimport)
#endif
#else
#define KUN_API __attribute__((visibility("default")))
#define KUN_EXPORT KUN_API
#endif
#pragma once

#include <immintrin.h>

namespace kun {
    struct Context;


    using f32x8 = __m256;
}

#ifdef _MSC_VER
#define KUN_API __declspec(dllexport)
#else
#define KUN_API __attribute__((visibility("default")))
#endif
#pragma once

#include <immintrin.h>

#ifdef __cplusplus
namespace kun {
struct Context;
using f32x8 = __m256;
static constexpr size_t simd_len = 8;
static constexpr size_t time_stride = 8;
} // namespace kun
#endif

#ifdef _MSC_VER
#define KUN_EXPORT extern "C" __declspec(dllexport)
#ifdef KUN_CORE_LIB
#define KUN_API __declspec(dllexport)
#else
#define KUN_API __declspec(dllimport)
#endif
#else
#define KUN_API __attribute__((visibility("default")))
#define KUN_EXPORT KUN_API
#endif
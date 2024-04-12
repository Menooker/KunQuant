#pragma once

#include <cstddef>
#ifdef __cplusplus
namespace kun {
struct Context;
static constexpr size_t time_stride = 8;
} // namespace kun
#endif

#ifdef _MSC_VER
#define KUN_EXPORT extern "C" __declspec(dllexport)
#ifdef KUN_CORE_LIB
#define KUN_API __declspec(dllexport)
#define KUN_TEMPLATE_EXPORT KUN_API
#else
#define KUN_API __declspec(dllimport)
#define KUN_TEMPLATE_EXPORT
#endif
#define KUN_TEMPLATE_ARG
#else
#define KUN_API __attribute__((visibility("default")))
#define KUN_EXPORT KUN_API
#define KUN_TEMPLATE_EXPORT KUN_API
// g++ has an strange behavior, it needs T to be
// exported if we want to export func<T>
#define KUN_TEMPLATE_ARG KUN_API
#endif
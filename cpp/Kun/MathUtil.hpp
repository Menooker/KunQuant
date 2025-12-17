#pragma once
#include <cstddef>

namespace kun {
namespace {
size_t divideAndCeil(size_t x, size_t y) { return (x + y - 1) / y; }
size_t roundUp(size_t x, size_t y) { return divideAndCeil(x, y) * y; }

} // namespace
} // namespace kun
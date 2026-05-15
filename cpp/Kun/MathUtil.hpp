#pragma once
#include <cstddef>

namespace kun {
namespace {
[[maybe_unused]] size_t divideAndCeil(size_t x, size_t y) {
    return (x + y - 1) / y;
}
[[maybe_unused]] size_t roundUp(size_t x, size_t y) {
    return divideAndCeil(x, y) * y;
}

} // namespace
} // namespace kun
#pragma once
#include "Stage.hpp"
#include <stddef.h>

namespace kun {
namespace ops {

struct MapperST8s {
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock) {
        auto S = stockid / simd_len;
        return S * num_time * simd_len + t * simd_len + stockid % simd_len;
    }
};

struct MapperTS {
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock) {
        return t * num_stock + stockid;
    }
};

} // namespace ops
} // namespace kun

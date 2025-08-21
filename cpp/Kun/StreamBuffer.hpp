#pragma once
#include "Base.hpp"
#include <cassert>
#include <stddef.h>

namespace kun {
namespace {
size_t divideAndCeil(size_t x, size_t y) { return (x + y - 1) / y; }
size_t roundUp(size_t x, size_t y) { return divideAndCeil(x, y) * y; }

} // namespace
template <typename T>
struct StreamBuffer {
    // [#stock_count of float data]
    // [#stock_count of float data]
    //   ... total window_size rows
    // [#stock_count of float data]
    // [stock_count/simd_len of Buffer positions (size_t)]

    // fix-me: we can store the pre-aligned stock_count to avoid re-computation
    // of roundUp
    alignas(64) char buf[0];
    T *getBuffer() const { return (T *)(buf); }
    size_t *getPos(size_t idx, size_t stock_count, size_t window_size) const {
        assert(stock_count % 4 == 0);
        return (size_t *)(buf + sizeof(T) * stock_count * window_size +
                          idx * sizeof(size_t));
    }
    static size_t getBufferSize(size_t stock_count, size_t window_size,
                                size_t simd_len) {
        return sizeof(T) * roundUp(stock_count, simd_len) * window_size +
               divideAndCeil(stock_count, simd_len) * sizeof(size_t);
    }
    static char *make(size_t stock_count, size_t window_size, size_t simd_len);
    const T *getCurrentBufferPtr(size_t stock_count, size_t window_size,
                                 size_t simd_len) const {
        stock_count = roundUp(stock_count, simd_len);
        size_t pos = *getPos(0, stock_count, window_size);
        size_t offset = 1;
        auto idx =
            pos >= offset ? (pos - offset) : (pos + window_size - offset);
        return getBuffer() + idx * stock_count;
    }
    T *pushData(size_t stock_count, size_t window_size, size_t simd_len) {
        stock_count = roundUp(stock_count, simd_len);
        size_t pos = *getPos(0, stock_count, window_size);
        auto ret = getBuffer() + pos * stock_count;
        pos += 1;
        pos = (pos >= window_size) ? 0 : pos;
        size_t *posbase = getPos(0, stock_count, window_size);
        for (int i = 0; i < divideAndCeil(stock_count, simd_len); i++) {
            posbase[i] = pos;
        }
        return ret;
    }
};

} // namespace kun
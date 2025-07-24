#pragma once
#include "Base.hpp"
#include <stddef.h>

namespace kun {
template<typename T>
struct StreamBuffer {
    // [#stock_count of float data]
    // [#stock_count of float data]
    //   ... total window_size rows
    // [#stock_count of float data]
    // [stock_count/simd_len of Buffer positions (size_t)]

    alignas(32) char buf[0];
    T *getBuffer() const { return (T *)(buf); }
    size_t *getPos(size_t idx, size_t stock_count, size_t window_size) const {
        return (size_t *)(buf + sizeof(T) * stock_count * window_size +
                          idx * sizeof(size_t));
    }
    static size_t getBufferSize(size_t stock_count, size_t window_size,
                                size_t simd_len) {
        return sizeof(T) * stock_count * window_size +
               stock_count / simd_len * sizeof(size_t);
    }
    static char *make(size_t stock_count, size_t window_size, size_t simd_len);
    const T *getCurrentBufferPtr(size_t stock_count,
                                     size_t window_size) const {
        size_t pos = *getPos(0, stock_count, window_size);
        size_t offset = 1;
        auto idx =
            pos >= offset ? (pos - offset) : (pos + window_size - offset);
        return getBuffer() + idx * stock_count;
    }
    T *pushData(size_t stock_count, size_t window_size, size_t simd_len) {
        size_t pos = *getPos(0, stock_count, window_size);
        auto ret = getBuffer() + pos * stock_count;
        pos += 1;
        pos = (pos >= window_size) ? 0 : pos;
        size_t *posbase = getPos(0, stock_count, window_size);
        for (int i = 0; i < stock_count / simd_len; i++) {
            posbase[i] = pos;
        }
        return ret;
    }
};

} // namespace kun
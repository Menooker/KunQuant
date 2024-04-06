#pragma once
#include "Context.hpp"
#include "Stage.hpp"
#include <stddef.h>

namespace kun {
namespace ops {

template<typename T, size_t simd_len>
struct KUN_API MapperSTs {
    static const T *getInput(Buffer *b, BufferInfo *info,
                                 size_t num_stock) {
        return b->getPtr<T>();
    }
    static T *getOutput(Buffer *b, BufferInfo *info, size_t num_stock,
                            size_t simd_len2) {
        return b->getPtr<T>();
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock, size_t simd_len2) {
        auto S = stockid / simd_len;
        return S * num_time * simd_len + t * simd_len + stockid % simd_len;
    }
};

template<typename T, size_t simd_len>
struct KUN_API MapperTS {
    static const T *getInput(Buffer *b, BufferInfo *info,
                                 size_t num_stock) {
        return b->getPtr<T>();
    }
    static T *getOutput(Buffer *b, BufferInfo *info, size_t num_stock,
                            size_t simd_len2) {
        return b->getPtr<T>();
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock, size_t simd_len2) {
        return t * num_stock + stockid;
    }
};

template<typename T, size_t simd_len>
struct KUN_API MapperSTREAM {
    static const T *getInput(Buffer *b, BufferInfo *info,
                                 size_t num_stock) {
        return b->stream_buf->getCurrentBufferPtr(num_stock, info->window);
    }
    static T *getOutput(Buffer *b, BufferInfo *info, size_t num_stock,
                            size_t simd_len2) {
        return b->stream_buf->pushData(num_stock, info->window, simd_len);
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock, size_t simd_len2) {
        return stockid;
    }
};

} // namespace ops
} // namespace kun

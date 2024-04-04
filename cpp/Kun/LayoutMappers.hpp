#pragma once
#include "Context.hpp"
#include "Stage.hpp"
#include <stddef.h>

namespace kun {
namespace ops {

struct MapperSTs {
    static const float *getInput(Buffer *b, BufferInfo *info,
                                 size_t num_stock) {
        return b->ptr;
    }
    static float *getOutput(Buffer *b, BufferInfo *info, size_t num_stock,
                            size_t simd_len) {
        return b->ptr;
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock, size_t simd_len) {
        auto S = stockid / simd_len;
        return S * num_time * simd_len + t * simd_len + stockid % simd_len;
    }
};

struct MapperTS {
    static const float *getInput(Buffer *b, BufferInfo *info,
                                 size_t num_stock) {
        return b->ptr;
    }
    static float *getOutput(Buffer *b, BufferInfo *info, size_t num_stock,
                            size_t simd_len) {
        return b->ptr;
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock, size_t simd_len) {
        return t * num_stock + stockid;
    }
};

struct MapperSTREAM {
    static const float *getInput(Buffer *b, BufferInfo *info,
                                 size_t num_stock) {
        return b->stream_buf->getCurrentBufferPtr(num_stock, info->window);
    }
    static float *getOutput(Buffer *b, BufferInfo *info, size_t num_stock,
                            size_t simd_len) {
        return b->stream_buf->pushData(num_stock, info->window, simd_len);
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock, size_t simd_len) {
        return stockid;
    }
};

} // namespace ops
} // namespace kun

#pragma once
#include "Context.hpp"
#include "Stage.hpp"
#include <stddef.h>
#include <type_traits>

namespace kun {
namespace ops {

template <typename T, size_t simd_len>
struct KUN_TEMPLATE_ARG MapperSTs {
    using type = T;
    static const T *getInput(Buffer *b, BufferInfo *info, size_t num_stock) {
        return b->getPtr<T>();
    }
    static T *getOutput(Buffer *b, BufferInfo *info, size_t num_stock) {
        return b->getPtr<T>();
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock) {
        auto S = stockid / simd_len;
        return S * num_time * simd_len + t * simd_len + stockid % simd_len;
    }
};

template <typename T, size_t simd_len>
struct KUN_TEMPLATE_ARG MapperTS {
    using type = T;
    static const T *getInput(Buffer *b, BufferInfo *info, size_t num_stock) {
        return b->getPtr<T>();
    }
    static T *getOutput(Buffer *b, BufferInfo *info, size_t num_stock) {
        return b->getPtr<T>();
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock) {
        return t * num_stock + stockid;
    }
};

template <typename T, size_t simd_len>
struct KUN_TEMPLATE_ARG MapperSTREAM {
    using type = T;
    static const T *getInput(Buffer *b, BufferInfo *info, size_t num_stock) {
        return b->getStream((T *)nullptr)
            ->getCurrentBufferPtr(num_stock, info->window, simd_len);
    }
    static T *getOutput(Buffer *b, BufferInfo *info, size_t num_stock) {
        return b->getStream((T *)nullptr)
            ->pushData(num_stock, info->window, simd_len);
    }
    static size_t call(size_t stockid, size_t t, size_t num_time,
                       size_t num_stock) {
        return stockid;
    }
};

namespace {
template <typename Mapper>
struct ExtractInputBuffer {
    using T = typename Mapper::type;
    using Ptr = const T *;
    static const T *get(RuntimeStage *stage, size_t buffer_idx,
                        size_t num_stock, size_t &num_time) {
        auto *buffer_info = stage->stage->in_buffers[buffer_idx];
        auto &inbuf = stage->ctx->buffers[buffer_info->id];
        num_time = inbuf.num_time;
        return Mapper::getInput(&inbuf, buffer_info, num_stock);
    }
};

template <typename Mapper>
struct ExtractOutputBuffer {
    using T = typename Mapper::type;
    using Ptr = T *;
    static T *get(RuntimeStage *stage, size_t buffer_idx, size_t num_stock,
                  size_t &num_time) {
        auto *buffer_info = stage->stage->out_buffers[buffer_idx];
        auto &inbuf = stage->ctx->buffers[buffer_info->id];
        num_time = inbuf.num_time;
        return Mapper::getOutput(&inbuf, buffer_info, num_stock);
    }
};

template <typename Mapper, template <typename> class ExtractBuffer>
struct CrossSectionalDataHolder {
    using T = typename Mapper::type;
    using Extract = ExtractBuffer<Mapper>;
    size_t num_stocks;
    typename Extract::Ptr buffer;
    size_t num_time;
    size_t base_time;

    CrossSectionalDataHolder(RuntimeStage *stage, size_t buffer_idx,
                             size_t __total_time, size_t __start) {
        num_stocks = stage->ctx->stock_count;
        base_time = (num_time == __total_time) ? __start : 0;
        buffer = Extract::get(stage, buffer_idx, num_stocks, num_time);
    }

    struct Accessor {
        CrossSectionalDataHolder &holder;
        size_t time_idx;

        Accessor(CrossSectionalDataHolder &holder, size_t time_idx)
            : holder(holder), time_idx(time_idx) {}

        auto operator[](size_t stockid) -> decltype(holder.buffer[0]) {
            return holder
                .buffer[Mapper::call(stockid, time_idx - holder.base_time,
                                     holder.num_time, holder.num_stocks)];
        }
    };

    Accessor accessor(size_t time_idx) { return Accessor(*this, time_idx); }
};

} // namespace

} // namespace ops
} // namespace kun


#include <Kun/Context.hpp>
#include <Kun/LayoutMappers.hpp>
#include <Kun/Module.hpp>
#include <Kun/Ops.hpp>
#include <Kun/RunGraph.hpp>

namespace kun {
namespace ops {

template <typename T, int simdlen>
static void aggregration(RuntimeStage *stage, size_t stock_idx,
                         size_t __total_time, size_t __start, size_t __length) {
    auto num_stock = stage->ctx->stock_count;
    auto &buffers = stage->ctx->buffers;
    auto &module_in_buffers = stage->stage->in_buffers;
    auto &module_out_buffers = stage->stage->out_buffers;
    auto &lablebuf = buffers[module_in_buffers[0]->id];
    auto &inbuf_orig = buffers[module_in_buffers[1]->id];

    auto &sumbuf_orig = buffers[module_out_buffers[AGGREGRATION_SUM]->id];
    auto &minbuf_orig = buffers[module_out_buffers[AGGREGRATION_MIN]->id];
    auto &maxbuf_orig = buffers[module_out_buffers[AGGREGRATION_MAX]->id];
    auto &firstbuf_orig = buffers[module_out_buffers[AGGREGRATION_FIRST]->id];
    auto &lastbuf_orig = buffers[module_out_buffers[AGGREGRATION_LAST]->id];
    auto &countbuf_orig = buffers[module_out_buffers[AGGREGRATION_COUNT]->id];
    auto &meanbuf_orig = buffers[module_out_buffers[AGGREGRATION_MEAN]->id];

    T *labels = lablebuf.getPtr<T>() + __start;
    InputTS<T, simdlen> inbuf{inbuf_orig.getPtr<T>(), stock_idx, num_stock,
                              __total_time, __start};

    OutputTS<T, simdlen> sumbuf{sumbuf_orig.getPtr<T>(), stock_idx, num_stock,
                                __total_time, 0};
    OutputTS<T, simdlen> minbuf{minbuf_orig.getPtr<T>(), stock_idx, num_stock,
                                __total_time, 0};
    OutputTS<T, simdlen> maxbuf{maxbuf_orig.getPtr<T>(), stock_idx, num_stock,
                                __total_time, 0};
    OutputTS<T, simdlen> firstbuf{firstbuf_orig.getPtr<T>(), stock_idx,
                                  num_stock, __total_time, 0};
    OutputTS<T, simdlen> lastbuf{lastbuf_orig.getPtr<T>(), stock_idx, num_stock,
                                 __total_time, 0};
    OutputTS<T, simdlen> countbuf{countbuf_orig.getPtr<T>(), stock_idx,
                                  num_stock, __total_time, 0};
    OutputTS<T, simdlen> meanbuf{meanbuf_orig.getPtr<T>(), stock_idx, num_stock,
                                 __total_time, 0};

    using SimdT = kun_simd::vec<T, simdlen>;
    ReduceMin<T, simdlen> reduce_min;
    ReduceMax<T, simdlen> reduce_max;
    ReduceAdd<T, simdlen> reduce_add;
    SimdT first = inbuf.step(0);
    SimdT last;
    SimdT count{0};

    auto todo_count = num_stock - stock_idx * simdlen;
    todo_count = todo_count > simdlen ? simdlen : todo_count;
    auto mask = SimdT::make_mask(todo_count);
    T last_label = labels[0];
    size_t store_idx = 0;
    for (size_t i = 0; i < __length; i++) {
        auto label = labels[i];
        auto cur = inbuf.step(i);
        if (label != last_label) {
            if (sumbuf.buf)
                sumbuf.store(store_idx, reduce_add, mask);
            if (minbuf.buf)
                minbuf.store(store_idx, reduce_min, mask);
            if (maxbuf.buf)
                maxbuf.store(store_idx, reduce_max, mask);
            if (firstbuf.buf)
                firstbuf.store(store_idx, first, mask);
            if (lastbuf.buf)
                lastbuf.store(store_idx, last, mask);
            if (countbuf.buf)
                countbuf.store(store_idx, count, mask);
            if (meanbuf.buf)
                meanbuf.store(store_idx, reduce_add / count, mask);
            store_idx++;
            first = cur;
            reduce_min = ReduceMin<T, simdlen>{};
            reduce_max = ReduceMax<T, simdlen>{};
            reduce_add = ReduceAdd<T, simdlen>{};
            count = T{0};
        }
        count = count + T{1};
        last = cur;
        last_label = label;
        reduce_max.step(cur, i);
        reduce_min.step(cur, i);
        reduce_add.step(cur, i);
    }
    if (sumbuf.buf)
        sumbuf.store(store_idx, reduce_add, mask);
    if (minbuf.buf)
        minbuf.store(store_idx, reduce_min, mask);
    if (maxbuf.buf)
        maxbuf.store(store_idx, reduce_max, mask);
    if (firstbuf.buf)
        firstbuf.store(store_idx, first, mask);
    if (lastbuf.buf)
        lastbuf.store(store_idx, last, mask);
    if (countbuf.buf)
        countbuf.store(store_idx, count, mask);
    if (meanbuf.buf)
        meanbuf.store(store_idx, reduce_add / count, mask);
}

void aggregrationFloat(RuntimeStage *stage, size_t stock_idx,
                       size_t __total_time, size_t __start, size_t __length) {
    aggregration<float, KUN_DEFAULT_FLOAT_SIMD_LEN>(
        stage, stock_idx, __total_time, __start, __length);
}

void aggregrationDouble(RuntimeStage *stage, size_t stock_idx,
                        size_t __total_time, size_t __start, size_t __length) {
    aggregration<double, KUN_DEFAULT_DOUBLE_SIMD_LEN>(
        stage, stock_idx, __total_time, __start, __length);
}

} // namespace ops
} // namespace kun
#include <Kun/Context.hpp>
#include <Kun/LayoutMappers.hpp>
#include <Kun/Module.hpp>
#include <Kun/Ops.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace kun {
namespace ops {

template <typename INPUT, typename OUTPUT>
static void ScaleStocks(RuntimeStage *stage, size_t time_idx,
                        size_t __total_time, size_t __start, size_t __length) {
    auto num_stocks = stage->ctx->stock_count;
    auto &inbuf = stage->ctx->buffers[stage->stage->in_buffers[0]->id];
    auto in_num_time = inbuf.num_time;
    auto in_base_time = (in_num_time == __total_time) ? 0 : __start;
    const float *input =
        INPUT::getInput(&inbuf, stage->stage->in_buffers[0], num_stocks);
    auto outinfo = stage->stage->out_buffers[0];
    auto simd_len = stage->ctx->simd_len;
    float *output = OUTPUT::getOutput(&stage->ctx->buffers[outinfo->id],
                                      outinfo, num_stocks, simd_len);
    auto time_end =
        std::min(__start + (time_idx + 1) * time_stride, __start + __length);
    for (size_t t = __start + time_idx * time_stride; t < time_end; t++) {
        float sum = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            float in = input[INPUT::call(i, t - in_base_time, in_num_time,
                                         num_stocks, simd_len)];
            if (!std::isnan(in)) {
                sum += std::abs(in);
            }
        }
        for (size_t i = 0; i < num_stocks; i++) {
            float in = input[INPUT::call(i, t - in_base_time, in_num_time,
                                         num_stocks, simd_len)];
            float out = (in == 0 && sum == 0) ? NAN : (in / sum);
            output[OUTPUT::call(i, t - __start, __length, num_stocks,
                                simd_len)] = out;
        }
    }
}

void ScaleStocksSTs_STs(RuntimeStage *stage, size_t time_idx,
                        size_t __total_time, size_t __start, size_t __length) {
    ScaleStocks<MapperSTs, MapperSTs>(stage, time_idx, __total_time, __start,
                                      __length);
}

void ScaleStocksSTs_TS(RuntimeStage *stage, size_t time_idx,
                       size_t __total_time, size_t __start, size_t __length) {
    ScaleStocks<MapperSTs, MapperTS>(stage, time_idx, __total_time, __start,
                                     __length);
}

void ScaleStocksTS_TS(RuntimeStage *stage, size_t time_idx, size_t __total_time,
                      size_t __start, size_t __length) {
    ScaleStocks<MapperTS, MapperTS>(stage, time_idx, __total_time, __start,
                                    __length);
}

void ScaleStocksSTREAM_STREAM(RuntimeStage *stage, size_t time_idx,
                              size_t __total_time, size_t __start,
                              size_t __length) {
    ScaleStocks<MapperSTREAM, MapperSTREAM>(stage, time_idx, __total_time,
                                            __start, __length);
}

void ScaleStocksTS_STs(RuntimeStage *stage, size_t time_idx,
                       size_t __total_time, size_t __start, size_t __length) {
    ScaleStocks<MapperTS, MapperSTs>(stage, time_idx, __total_time, __start,
                                     __length);
}

} // namespace ops
} // namespace kun
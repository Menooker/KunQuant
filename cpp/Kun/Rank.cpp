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
static void RankStocks(RuntimeStage *stage, size_t time_idx,
                       size_t __total_time, size_t __start, size_t __length) {
    auto num_stocks = stage->ctx->stock_count;
    auto &inbuf = stage->ctx->buffers[stage->stage->in_buffers[0]->id];
    auto in_num_time = inbuf.num_time;
    auto in_base_time = (in_num_time == __total_time) ? 0 : __start;
    const float *input =
        INPUT::getInput(&inbuf, stage->stage->in_buffers[0], num_stocks);
    auto outinfo = stage->stage->out_buffers[0];
    float *output = OUTPUT::getOutput(&stage->ctx->buffers[outinfo->id], outinfo,
                                      num_stocks);
    auto time_end =
        std::min(__start + (time_idx + 1) * time_stride, __start + __length);
    std::vector<float> data;
    data.reserve(num_stocks);
    for (size_t t = __start + time_idx * time_stride; t < time_end; t++) {
        for (size_t i = 0; i < num_stocks; i++) {
            auto S = i / simd_len;
            float in = input[INPUT::call(i, t - in_base_time, in_num_time,
                                         num_stocks)];
            if (!std::isnan(in)) {
                data.push_back(in);
            }
        }
        std::sort(data.begin(), data.end());
        for (size_t i = 0; i < num_stocks; i++) {
            auto S = i / simd_len;
            float in = input[INPUT::call(i, t - in_base_time, in_num_time,
                                         num_stocks)];
            float out;
            if (!std::isnan(in)) {
                auto pos = std::equal_range(data.begin(), data.end(), in);
                auto start = pos.first - data.begin();
                auto end = pos.second - data.begin();
                out = ((start + end - 1) / 2.0f + 1.0f) / data.size();
            } else {
                out = NAN;
            }
            output[OUTPUT::call(i, t - __start, __length, num_stocks)] = out;
        }
        data.clear();
    }
}

void RankStocksST8s_ST8s(RuntimeStage *stage, size_t time_idx,
                         size_t __total_time, size_t __start, size_t __length) {
    RankStocks<MapperST8s, MapperST8s>(stage, time_idx, __total_time, __start,
                                       __length);
}

void RankStocksST8s_TS(RuntimeStage *stage, size_t time_idx,
                       size_t __total_time, size_t __start, size_t __length) {
    RankStocks<MapperST8s, MapperTS>(stage, time_idx, __total_time, __start,
                                     __length);
}

void RankStocksTS_TS(RuntimeStage *stage, size_t time_idx, size_t __total_time,
                     size_t __start, size_t __length) {
    RankStocks<MapperTS, MapperTS>(stage, time_idx, __total_time, __start,
                                   __length);
}

void RankStocksSTREAM_STREAM(RuntimeStage *stage, size_t time_idx,
                             size_t __total_time, size_t __start,
                             size_t __length) {
    RankStocks<MapperSTREAM, MapperSTREAM>(stage, time_idx, __total_time,
                                           __start, __length);
}

void RankStocksTS_ST8s(RuntimeStage *stage, size_t time_idx,
                       size_t __total_time, size_t __start, size_t __length) {
    RankStocks<MapperTS, MapperST8s>(stage, time_idx, __total_time, __start,
                                     __length);
}

} // namespace ops
} // namespace kun
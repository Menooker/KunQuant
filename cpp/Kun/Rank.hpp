#pragma once

#include <Kun/Context.hpp>
#include <Kun/LayoutMappers.hpp>
#include <Kun/Module.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace kun {
namespace ops {
template <typename INPUT, typename OUTPUT>
void KUN_TEMPLATE_EXPORT RankStocks(RuntimeStage *stage, size_t time_idx,
                                    size_t __total_time, size_t __start,
                                    size_t __length) {
    auto num_stocks = stage->ctx->stock_count;
    auto &inbuf = stage->ctx->buffers[stage->stage->in_buffers[0]->id];
    auto in_num_time = inbuf.num_time;
    auto in_base_time = (in_num_time == __total_time) ? 0 : __start;
    const auto *input =
        INPUT::getInput(&inbuf, stage->stage->in_buffers[0], num_stocks);
    using T = typename std::decay<decltype(*input)>::type;
    auto outinfo = stage->stage->out_buffers[0];
    T *output = OUTPUT::getOutput(&stage->ctx->buffers[outinfo->id], outinfo,
                                  num_stocks);
    auto time_end =
        std::min(__start + (time_idx + 1) * time_stride, __start + __length);
    std::vector<T> data;
    data.reserve(num_stocks);
    for (size_t t = __start + time_idx * time_stride; t < time_end; t++) {
        for (size_t i = 0; i < num_stocks; i++) {
            T in = input[INPUT::call(i, t - in_base_time, in_num_time,
                                     num_stocks)];
            if (!std::isnan(in)) {
                data.push_back(in);
            }
        }
        std::sort(data.begin(), data.end());
        for (size_t i = 0; i < num_stocks; i++) {
            T in = input[INPUT::call(i, t - in_base_time, in_num_time,
                                     num_stocks)];
            T out;
            if (!std::isnan(in)) {
                auto pos = std::equal_range(data.begin(), data.end(), in);
                auto start = pos.first - data.begin();
                auto end = pos.second - data.begin();
                auto sum = (start + end + 1) * (end - start) / 2;
                out = T(sum) / T(end - start) / T(data.size());
                // out = ((start + end - 1) / T{2.0} + T{1.0}) / data.size();
            } else {
                out = NAN;
            }
            output[OUTPUT::call(i, t - __start, __length, num_stocks)] = out;
        }
        data.clear();
    }
}

extern template void RankStocks<MapperSTs<float, 8>, MapperSTs<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
extern template void RankStocks<MapperSTs<float, 8>, MapperTS<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
extern template void RankStocks<MapperTS<float, 8>, MapperTS<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
extern template void RankStocks<MapperTS<float, 8>, MapperSTs<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
extern template void RankStocks<MapperSTREAM<float, 8>, MapperSTREAM<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);

} // namespace ops
} // namespace kun
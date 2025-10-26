#pragma once

#include <Kun/Context.hpp>
#include <Kun/LayoutMappers.hpp>
#include <Kun/Module.hpp>
#include <algorithm>
#include <cmath>

namespace kun {
namespace ops {
template <typename INPUT>
void KUN_TEMPLATE_EXPORT CorrWith(RuntimeStage *stage, size_t time_idx,
                                  size_t __total_time, size_t __start,
                                  size_t __length) {
    auto num_stocks = stage->ctx->stock_count;
    auto &inbuf0 = stage->ctx->buffers[stage->stage->in_buffers[0]->id];
    auto in_num_time0 = inbuf0.num_time;
    auto in_base_time0 = (in_num_time0 == __total_time) ? 0 : __start;
    const auto *input0 =
        INPUT::getInput(&inbuf0, stage->stage->in_buffers[0], num_stocks);

    auto &inbuf1 = stage->ctx->buffers[stage->stage->in_buffers[1]->id];
    auto in_num_time1 = inbuf1.num_time;
    auto in_base_time1 = (in_num_time1 == __total_time) ? 0 : __start;
    const auto *input1 =
        INPUT::getInput(&inbuf1, stage->stage->in_buffers[1], num_stocks);
    using T = typename std::decay<decltype(*input0)>::type;
    auto outinfo = stage->stage->out_buffers[0];
    auto simd_len = stage->ctx->simd_len;
    T *output = stage->ctx->buffers[outinfo->id].getPtr<T>();
    auto time_end =
        std::min(__start + (time_idx + 1) * time_stride, __start + __length);
    for (size_t t = __start + time_idx * time_stride; t < time_end; t++) {
        T XY = 0, X = 0, Y = 0, X2 = 0, Y2 = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T x = input0[INPUT::call(i, t - in_base_time0, in_num_time0,
                                     num_stocks)];
            T y = input1[INPUT::call(i, t - in_base_time1, in_num_time1,
                                     num_stocks)];
            X += x;
            Y += y;
            X2 += x * x;
            Y2 += y * y;
            XY += x * y;
        }
        XY /= num_stocks;
        X /= num_stocks;
        Y /= num_stocks;
        X2 /= num_stocks;
        Y2 /= num_stocks;
        output[t - __start] =
            (XY - X * Y) / (std::sqrt(X2 - X * X) * std::sqrt(Y2 - Y * Y));
    }
}

template <typename INPUT>
void KUN_TEMPLATE_EXPORT RankCorrWith(RuntimeStage *stage, size_t time_idx,
                                      size_t __total_time, size_t __start,
                                      size_t __length) {
    auto num_stocks = stage->ctx->stock_count;
    auto &inbuf0 = stage->ctx->buffers[stage->stage->in_buffers[0]->id];
    auto in_num_time0 = inbuf0.num_time;
    auto in_base_time0 = (in_num_time0 == __total_time) ? 0 : __start;
    const auto *input0 =
        INPUT::getInput(&inbuf0, stage->stage->in_buffers[0], num_stocks);

    auto &inbuf1 = stage->ctx->buffers[stage->stage->in_buffers[1]->id];
    auto in_num_time1 = inbuf1.num_time;
    auto in_base_time1 = (in_num_time1 == __total_time) ? 0 : __start;
    const auto *input1 =
        INPUT::getInput(&inbuf1, stage->stage->in_buffers[1], num_stocks);
    using T = typename std::decay<decltype(*input0)>::type;
    auto outinfo = stage->stage->out_buffers[0];
    auto simd_len = stage->ctx->simd_len;
    T *output = stage->ctx->buffers[outinfo->id].getPtr<T>();
    auto time_end =
        std::min(__start + (time_idx + 1) * time_stride, __start + __length);
    std::vector<T> data;
    data.reserve(num_stocks);
    for (size_t t = __start + time_idx * time_stride; t < time_end; t++) {
        for (size_t i = 0; i < num_stocks; i++) {
            T in = input0[INPUT::call(i, t - in_base_time0, in_num_time0,
                                      num_stocks)];
            if (!std::isnan(in)) {
                data.push_back(in);
            }
        }
        std::sort(data.begin(), data.end());
        T XY = 0, X = 0, Y = 0, X2 = 0, Y2 = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T in = input0[INPUT::call(i, t - in_base_time0, in_num_time0,
                                      num_stocks)];
            T x;
            if (!std::isnan(in)) {
                auto pos = std::equal_range(data.begin(), data.end(), in);
                auto start = pos.first - data.begin();
                auto end = pos.second - data.begin();
                auto sum = (start + end + 1) * (end - start) / 2;
                x = T(sum) / T(end - start) / T(data.size());
                // out = ((start + end - 1) / T{2.0} + T{1.0}) / data.size();
            } else {
                x = NAN;
            }
            T y = input1[INPUT::call(i, t - in_base_time1, in_num_time1,
                                     num_stocks)];
            X += x;
            Y += y;
            X2 += x * x;
            Y2 += y * y;
            XY += x * y;
        }
        data.clear();
        XY /= num_stocks;
        X /= num_stocks;
        Y /= num_stocks;
        X2 /= num_stocks;
        Y2 /= num_stocks;
        output[t - __start] =
            (XY - X * Y) / (std::sqrt(X2 - X * X) * std::sqrt(Y2 - Y * Y));
    }
}

extern template void CorrWith<MapperSTs<float, 8>>(RuntimeStage *stage,
                                                   size_t time_idx,
                                                   size_t __total_time,
                                                   size_t __start,
                                                   size_t __length);
extern template void CorrWith<MapperTS<float, 8>>(RuntimeStage *stage,
                                                  size_t time_idx,
                                                  size_t __total_time,
                                                  size_t __start,
                                                  size_t __length);
extern template void RankCorrWith<MapperSTs<float, 8>>(RuntimeStage *stage,
                                                       size_t time_idx,
                                                       size_t __total_time,
                                                       size_t __start,
                                                       size_t __length);
extern template void RankCorrWith<MapperTS<float, 8>>(RuntimeStage *stage,
                                                      size_t time_idx,
                                                      size_t __total_time,
                                                      size_t __start,
                                                      size_t __length);

} // namespace ops
} // namespace kun
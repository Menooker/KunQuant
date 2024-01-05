#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/Ops.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace kun {
namespace ops {

void RankST8sTimeStride8(RuntimeStage *stage, size_t __stock_idx,
                         size_t __total_time, size_t __start, size_t __length) {
    auto num_stocks = stage->ctx->stock_count;
    auto num_time = stage->ctx->total_time;
    float* input = stage->ctx->buffers[stage->stage->in_buffers[0]->id].ptr;
    float* output = stage->ctx->buffers[stage->stage->out_buffers[0]->id].ptr;
    auto time_end =
        std::min(__start + (__stock_idx + 1) * time_stride, __start + __length);
    std::vector<float> data;
    data.reserve(num_stocks);
    for (size_t t = __start + __stock_idx * time_stride; t < time_end; t++) {
        for (size_t i = 0; i < num_stocks; i++) {
            auto S = i / simd_len;
            float in = input[S * num_time * simd_len + t * simd_len + i % simd_len];
            if(!std::isnan(in)) {
                data.push_back(in);
            }
        }
        std::sort(data.begin(), data.end());
        for (size_t i = 0; i < num_stocks; i++) {
            auto S = i / simd_len;
            float in = input[S * num_time * simd_len + t * simd_len + i % simd_len];
            float out;
            if(!std::isnan(in)) {
                auto pos = std::equal_range(data.begin(), data.end(), in);
                auto start = pos.first - data.begin();
                auto end = pos.second - data.begin();
                out = ((start+end-1)/2.0f + 1.0f)/data.size(); 
            } else {
                out = NAN;
            }
            output[S * num_time * simd_len + t * simd_len + i % simd_len] = out;
        }
        data.clear();
    }
}
} // namespace ops
} // namespace kun
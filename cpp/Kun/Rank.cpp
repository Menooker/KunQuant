#include <Kun/Context.hpp>
#include <Kun/Module.hpp>
#include <Kun/Ops.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace kun {
namespace ops {

struct MapperST8s {
    static size_t call(size_t stockid, size_t t, size_t num_time, size_t num_stock) {
        auto S = stockid / simd_len;
        return S * num_time * simd_len + t * simd_len + stockid % simd_len;
    }
};

struct MapperTS {
    static size_t call(size_t stockid, size_t t, size_t num_time, size_t num_stock) {
        return t * num_stock + stockid;
    }
};

template <typename INPUT, typename OUTPUT>
static void RankStocks(RuntimeStage *stage, size_t time_idx,
                       size_t __total_time, size_t __start, size_t __length) {
    auto num_stocks = stage->ctx->stock_count;
    auto num_time = stage->ctx->total_time;
    float *input = stage->ctx->buffers[stage->stage->in_buffers[0]->id].ptr;
    float *output = stage->ctx->buffers[stage->stage->out_buffers[0]->id].ptr;
    auto time_end =
        std::min(__start + (time_idx + 1) * time_stride, __start + __length);
    std::vector<float> data;
    data.reserve(num_stocks);
    for (size_t t = __start + time_idx * time_stride; t < time_end; t++) {
        for (size_t i = 0; i < num_stocks; i++) {
            auto S = i / simd_len;
            float in = input[INPUT::call(i, t, num_time, num_stocks)];
            if (!std::isnan(in)) {
                data.push_back(in);
            }
        }
        std::sort(data.begin(), data.end());
        for (size_t i = 0; i < num_stocks; i++) {
            auto S = i / simd_len;
            float in = input[INPUT::call(i, t, num_time, num_stocks)];
            float out;
            if (!std::isnan(in)) {
                auto pos = std::equal_range(data.begin(), data.end(), in);
                auto start = pos.first - data.begin();
                auto end = pos.second - data.begin();
                out = ((start + end - 1) / 2.0f + 1.0f) / data.size();
            } else {
                out = NAN;
            }
            output[OUTPUT::call(i, t, num_time, num_stocks)] = out;
        }
        data.clear();
    }
}

void RankStocksST8s_ST8s(RuntimeStage *stage, size_t time_idx,
                         size_t __total_time, size_t __start, size_t __length) {
    RankStocks<MapperST8s, MapperST8s>(stage, time_idx, __total_time,
                                       __start, __length);
}

void RankStocksST8s_TS(RuntimeStage *stage, size_t time_idx,
                         size_t __total_time, size_t __start, size_t __length) {
    RankStocks<MapperST8s, MapperTS>(stage, time_idx, __total_time,
                                       __start, __length);
}

void RankStocksTS_TS(RuntimeStage *stage, size_t time_idx,
                         size_t __total_time, size_t __start, size_t __length) {
    RankStocks<MapperTS, MapperTS>(stage, time_idx, __total_time,
                                       __start, __length);
}

void RankStocksTS_ST8s(RuntimeStage *stage, size_t time_idx,
                         size_t __total_time, size_t __start, size_t __length) {
    RankStocks<MapperTS, MapperST8s>(stage, time_idx, __total_time,
                                       __start, __length);
}

} // namespace ops
} // namespace kun
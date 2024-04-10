#include "Rank.hpp"

namespace kun {
namespace ops {

template void RankStocks<MapperSTs<float, 8>, MapperSTs<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
template void RankStocks<MapperSTs<float, 8>, MapperTS<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
template void RankStocks<MapperTS<float, 8>, MapperTS<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
template void RankStocks<MapperTS<float, 8>, MapperSTs<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);
template void RankStocks<MapperSTREAM<float, 8>, MapperSTREAM<float, 8>>(
    RuntimeStage *stage, size_t time_idx, size_t __total_time, size_t __start,
    size_t __length);

} // namespace ops
} // namespace kun
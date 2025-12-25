#include "Rank.hpp"

namespace kun {
namespace ops {

#define DEF_INSTANCE(...)                                                      \
    template KUN_TEMPLATE_EXPORT void RankStocks<__VA_ARGS__>(                 \
        RuntimeStage * stage, size_t time_idx, size_t __total_time,            \
        size_t __start, size_t __length);

DEF_INSTANCE(MapperSTsFloat, MapperSTsFloat)
DEF_INSTANCE(MapperSTsFloat, MapperTSFloat)
DEF_INSTANCE(MapperTSFloat, MapperTSFloat)
DEF_INSTANCE(MapperTSFloat, MapperSTsFloat)
DEF_INSTANCE(MapperSTREAMFloat, MapperSTREAMFloat)

} // namespace ops
} // namespace kun
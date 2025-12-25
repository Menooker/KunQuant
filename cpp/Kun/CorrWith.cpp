#include "CorrWith.hpp"

namespace kun {
namespace ops {

#define DEF_INSTANCE(X, ...)                                                      \
    template KUN_TEMPLATE_EXPORT void X<__VA_ARGS__>(                 \
        RuntimeStage * stage, size_t time_idx, size_t __total_time,            \
        size_t __start, size_t __length);

DEF_INSTANCE(CorrWith, MapperSTsFloat)
DEF_INSTANCE(CorrWith, MapperTSFloat)
DEF_INSTANCE(RankCorrWith, MapperSTsFloat)
DEF_INSTANCE(RankCorrWith, MapperTSFloat)

} // namespace ops
} // namespace kun
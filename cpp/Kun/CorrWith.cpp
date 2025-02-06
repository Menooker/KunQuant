#include "CorrWith.hpp"

namespace kun {
namespace ops {

#define DEF_INSTANCE(X, ...)                                                      \
    template KUN_TEMPLATE_EXPORT void X<__VA_ARGS__>(                 \
        RuntimeStage * stage, size_t time_idx, size_t __total_time,            \
        size_t __start, size_t __length);

DEF_INSTANCE(CorrWith, MapperSTs<float, 8>)
DEF_INSTANCE(CorrWith, MapperTS<float, 8>)
DEF_INSTANCE(RankCorrWith, MapperSTs<float, 8>)
DEF_INSTANCE(RankCorrWith, MapperTS<float, 8>)

} // namespace ops
} // namespace kun
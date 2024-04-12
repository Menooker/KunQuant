#include "Scale.hpp"

namespace kun {
namespace ops {
#define DEF_INSTANCE(...)                                                      \
    template KUN_TEMPLATE_EXPORT void ScaleStocks<__VA_ARGS__>(                \
        RuntimeStage * stage, size_t time_idx, size_t __total_time,            \
        size_t __start, size_t __length);

DEF_INSTANCE(MapperSTs<float, 8>, MapperSTs<float, 8>)
DEF_INSTANCE(MapperSTs<float, 8>, MapperTS<float, 8>)
DEF_INSTANCE(MapperTS<float, 8>, MapperTS<float, 8>)
DEF_INSTANCE(MapperTS<float, 8>, MapperSTs<float, 8>)
DEF_INSTANCE(MapperSTREAM<float, 8>, MapperSTREAM<float, 8>)

} // namespace ops
} // namespace kun
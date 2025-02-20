#pragma once

#include "Base.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace kun {

struct RuntimeStage;
using FuncType = void (*)(Context *__ctx, size_t __stock_idx,
                          size_t __total_time, size_t __start, size_t __length);
using RankFuncType = void (*)(RuntimeStage *stage, size_t __stock_idx,
                         size_t __total_time, size_t __start, size_t __length);

enum class BufferKind: int32_t {
    INPUT = 0,
    OUTPUT,
    TEMP,
};


struct BufferInfo {
    size_t id;
    const char* name;
    size_t num_users;
    BufferKind kind;
    uint32_t unreliable_count;
    // the max window size of the ops depending on this buffer
    uint32_t window;
};

enum class TaskExecKind {
    SLICE_BY_STOCK,
    SLICE_BY_TIME,
};

union FuncHolder
{
    RankFuncType rankf;
    FuncType f;
    constexpr FuncHolder(RankFuncType r): rankf{r} {}
    constexpr FuncHolder(FuncType r): f{r} {}
};


enum class Datatype {
    Float,
    Double,
};


struct Stage {
    FuncHolder f;
    Stage **dependers;
    size_t num_dependers;
    BufferInfo **in_buffers;
    size_t num_in_buffers;
    BufferInfo **out_buffers;
    size_t num_out_buffers;
    size_t orig_pending;
    TaskExecKind kind;
    size_t id;
    // Stage(FuncType f, Stage **dependers, size_t num_dependers,
    //       size_t *in_buffers, size_t num_in_buffers, size_t *out_buffers,
    //       size_t num_out_buffers, size_t orig_pending)
    //     : f{f}, dependers{dependers}, num_dependers{num_dependers},
    //       in_buffers{in_buffers}, num_in_buffers{num_in_buffers},
    //       out_buffers{out_buffers}, num_out_buffers{num_out_buffers},
    //       orig_pending{orig_pending} {
    //       }
};

} // namespace kun
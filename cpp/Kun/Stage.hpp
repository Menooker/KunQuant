#pragma once

#include "Base.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace kun {

using FuncType = void (*)(Context *__ctx, size_t __stock_idx,
                          size_t __total_time, size_t __start, size_t __length);


enum class BufferKind: int32_t {
    INPUT = 0,
    OUTPUT,
    TEMP,
};


struct BufferInfo {
    size_t id;
    const char* name;
    BufferKind kind;
};

struct Stage {
    FuncType f;
    Stage **dependers;
    size_t num_dependers;
    BufferInfo **in_buffers;
    size_t num_in_buffers;
    BufferInfo **out_buffers;
    size_t num_out_buffers;
    size_t orig_pending;
    size_t num_tasks;
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
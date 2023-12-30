#pragma once

#include "Stage.hpp"

namespace kun {

enum class BufferKind: int32_t {
    INPUT = 0,
    OUTPUT,
    TEMP,
};

struct BufferInfo {
    const char* name;
    BufferKind kind;
};

struct Module {
    size_t num_stages;
    Stage* stages;
    size_t num_buffers;
    BufferInfo* buffers;
};

}
#pragma once

#include <immintrin.h>

namespace kun {
    struct Context {
        float** buffers;
    };


    using f32x8 = __m256;
}
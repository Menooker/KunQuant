#pragma once
#include <stddef.h>

namespace kun {
struct StreamBuffer {
    alignas(32) char buf[0];
    float *getBuffer() const {
        return (float *)(buf);
    }
    size_t *getPos(size_t idx, size_t stock_count,
                      size_t window_size) const {
        return (size_t *)(buf + sizeof(float) * stock_count * window_size +
                          idx * sizeof(size_t));
    }
};

} // namespace kun
#pragma once

#include "Base.hpp"
#include <cmath>
#include <stdint.h>
#include <type_traits>
#include <limits>

namespace kun {
namespace ops {

template <bool vcontainsWindow>
struct DataSource {
    constexpr static bool containsWindow = vcontainsWindow;
};

struct dummy {};

template <int stride>
struct Input : DataSource<true> {
    float *buf;
    f32x8 step(size_t index) { return _mm256_load_ps(&buf[index * stride]); }

    f32x8 getWindow(size_t index, size_t offset) {
        return _mm256_load_ps(&buf[(index - offset) * stride]);
    }

    f32x8 getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

template <int stride>
struct Output : DataSource<true> {
    float *buf;
    void store(size_t index, const f32x8 &v) {
        _mm256_store_ps(&buf[index * stride], v);
    }

    f32x8 getWindow(size_t index, size_t offset) {
        return _mm256_load_ps(&buf[(index - offset) * stride]);
    }
    f32x8 getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

template <size_t window>
struct OutputWindow : DataSource<true> {
    constexpr static size_t stride = 8;
    // window slots of floatx8
    alignas(64) float buf[window * stride];
    // next writable position
    size_t pos;
    OutputWindow() : pos{0} {
        for (size_t i = 0; i < window * stride; i++) {
            buf[i] = NAN;
        }
    }
    void store(size_t index, const f32x8 &in) {
        _mm256_store_ps(&buf[pos * stride], in);
        pos += 1;
        pos = (pos >= window) ? 0 : pos;
    }
    f32x8 getWindow(size_t index, size_t offset) {
        offset += 1;
        auto idx = pos >= offset ? (pos - offset) : (pos + window - offset);
        return _mm256_load_ps(&buf[idx * stride]);
    }
    f32x8 getWindowUnordered(size_t index, size_t offset) {
        return _mm256_load_ps(&buf[offset * stride]);
    }
};

template <typename TInput>
struct RequireWindow {
    static_assert(TInput::containsWindow, "This stage needs window data");
};

template <int window>
struct FastWindowedSum {
    f32x8 v = _mm256_setzero_ps();
    template <typename TInput>
    f32x8 step(TInput &input, f32x8 cur, size_t index) {
        RequireWindow<TInput>;
        if (index >= window) {
            v = _mm256_sub_ps(v, input.getWindow(index, window));
        }
        v = _mm256_add_ps(v, cur);
        if (index >= window) {
            return v;
        }
        return _mm256_set1_ps(NAN);
    }
};

struct ReduceAdd {
    f32x8 v = _mm256_setzero_ps();
    void step(f32x8 input, size_t index) {
        v= _mm256_add_ps(v, input);
    }
    operator f32x8() {
        return v;
    }
};

struct ReduceMin {
    f32x8 v = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    void step(f32x8 input, size_t index) {
        v= _mm256_min_ps(v, input);
    }
    operator f32x8() {
        return v;
    }
};

struct ReduceMax {
    f32x8 v = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    void step(f32x8 input, size_t index) {
        v= _mm256_max_ps(v, input);
    }
    operator f32x8() {
        return v;
    }
};


template <int window, typename TInput>
f32x8 windowedRef(TInput &input, size_t index) {
    RequireWindow<TInput>;
    if (index >= window) {
        return input.getWindow(index, window);
    }
    return _mm256_set1_ps(NAN);
}

inline f32x8 add(f32x8 a, f32x8 b) { return _mm256_add_ps(a, b); }
inline f32x8 add(f32x8 a, float b) {
    return _mm256_add_ps(a, _mm256_set1_ps(b));
}
inline f32x8 sub(f32x8 a, f32x8 b) { return _mm256_sub_ps(a, b); }
inline f32x8 sub(f32x8 a, float b) {
    return _mm256_sub_ps(a, _mm256_set1_ps(b));
}
inline f32x8 mul(f32x8 a, f32x8 b) { return _mm256_mul_ps(a, b); }
inline f32x8 mul(f32x8 a, float b) {
    return _mm256_mul_ps(a, _mm256_set1_ps(b));
}
inline f32x8 div(f32x8 a, f32x8 b) { return _mm256_div_ps(a, b); }
inline f32x8 div(f32x8 a, float b) {
    return _mm256_div_ps(a, _mm256_set1_ps(b));
}

inline f32x8 sqrt(f32x8 a) {
    return _mm256_sqrt_ps(a);
}

} // namespace stage
} // namespace kun
#pragma once

#include "Base.hpp"
#include "Math.hpp"
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>

namespace kun {
namespace ops {

template <bool vcontainsWindow>
struct DataSource {
    constexpr static bool containsWindow = vcontainsWindow;
};

struct dummy {};

inline f32x8 Select(f32x8 cond, f32x8 vtrue, f32x8 vfalse) {
    return _mm256_blendv_ps(vfalse, vtrue, cond);
}

struct InputST8s : DataSource<true> {
    constexpr static size_t stride = 8;
    float *__restrict buf;
    InputST8s(float *base, size_t stock_idx, size_t total_time, size_t start)
        : buf{base + stock_idx * total_time * stride + start * stride} {}
    f32x8 step(size_t index) { return _mm256_loadu_ps(&buf[index * stride]); }

    f32x8 getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return _mm256_set1_ps(NAN);
        }
        return _mm256_loadu_ps(&buf[(index - offset) * stride]);
    }

    f32x8 getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

struct OutputST8s : DataSource<true> {
    constexpr static size_t stride = 8;
    float *__restrict buf;
    OutputST8s(float *base, size_t stock_idx, size_t num_stock, size_t length,
               size_t start)
        : buf{base + stock_idx * length * stride} {}
    void store(size_t index, const f32x8 &v) {
        _mm256_storeu_ps(&buf[index * stride], v);
    }

    f32x8 getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return _mm256_set1_ps(NAN);
        }
        return _mm256_loadu_ps(&buf[(index - offset) * stride]);
    }
    f32x8 getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

struct OutputTS : DataSource<true> {
    constexpr static size_t stride = 8;
    float *__restrict buf;
    size_t stock_idx;
    size_t num_stock;
    OutputTS(float *base, size_t stock_idx, size_t num_stock, size_t length,
             size_t start)
        : buf{base}, stock_idx{stock_idx}, num_stock{num_stock} {}

    float *getPtr(size_t index) {
        return &buf[index * num_stock + stride * stock_idx];
    }
    void store(size_t index, const f32x8 &v) {
        _mm256_storeu_ps(getPtr(index), v);
    }

    f32x8 getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return _mm256_set1_ps(NAN);
        }
        return _mm256_loadu_ps(getPtr(index - offset));
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

static inline __m256i blendvps_si256(__m256i a, __m256i b, __m256i mask) {
    __m256 res =
        _mm256_blendv_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b),
                         _mm256_castsi256_ps(mask));
    return _mm256_castps_si256(res);
}

template <typename TInput>
struct RequireWindow {
    static_assert(TInput::containsWindow, "This stage needs window data");
};

inline f32x8 isNAN(f32x8 v) { return _mm256_cmp_ps(v, v, _CMP_UNORD_Q); }

// returns true when v1 or v2 is NAN
inline f32x8 isNAN(f32x8 v1, f32x8 v2) {
    return _mm256_cmp_ps(v1, v2, _CMP_UNORD_Q);
}

template <int window>
struct FastWindowedSum {
    f32x8 v = _mm256_setzero_ps();
    __m256i num_nans = _mm256_set1_epi32(window);
    template <typename TInput>
    f32x8 step(TInput &input, f32x8 cur, size_t index) {
        RequireWindow<TInput>{};
        auto old = input.getWindow(index, window);
        auto old_is_nan = isNAN(old);
        auto new_is_nan = isNAN(cur);
        // v = old_is_nan? v : (v-old)
        v = Select(old_is_nan, v, _mm256_sub_ps(v, old));
        // v = new_is_nan? v : (v-cur)
        v = Select(new_is_nan, v, _mm256_add_ps(v, cur));
        num_nans = _mm256_sub_epi32(
            num_nans,
            blendvps_si256(_mm256_setzero_si256(), _mm256_set1_epi32(1),
                           _mm256_castps_si256(old_is_nan)));
        num_nans = _mm256_add_epi32(
            num_nans,
            blendvps_si256(_mm256_setzero_si256(), _mm256_set1_epi32(1),
                           _mm256_castps_si256(new_is_nan)));
        auto out_is_normal = _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(num_nans, _mm256_setzero_si256()));
        return Select(out_is_normal, v, _mm256_set1_ps(NAN));
    }
};

struct ReduceAdd {
    f32x8 v = _mm256_setzero_ps();
    void step(f32x8 input, size_t index) { v = _mm256_add_ps(v, input); }
    operator f32x8() { return v; }
};

struct ReduceMin {
    f32x8 v = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    void step(f32x8 input, size_t index) { v = _mm256_min_ps(v, input); }
    operator f32x8() { return v; }
};

struct ReduceMax {
    f32x8 v = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    void step(f32x8 input, size_t index) { v = _mm256_max_ps(v, input); }
    operator f32x8() { return v; }
};

template <int window, typename TInput>
f32x8 windowedRef(TInput &input, size_t index) {
    RequireWindow<TInput>{};
    if (index >= window) {
        return input.getWindow(index, window);
    }
    return _mm256_set1_ps(NAN);
}

inline f32x8 LessThan(f32x8 a, f32x8 b) {
    return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}

inline f32x8 LessEqual(f32x8 a, f32x8 b) {
    return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}

inline f32x8 GreaterThan(f32x8 a, f32x8 b) {
    return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}

inline f32x8 GreaterEqual(f32x8 a, f32x8 b) {
    return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
}

inline f32x8 Equals(f32x8 a, f32x8 b) {
    return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}

inline f32x8 LessThan(f32x8 a, float b) {
    return _mm256_cmp_ps(a, _mm256_set1_ps(b), _CMP_LT_OQ);
}
inline f32x8 LessThanOrNan(f32x8 a, f32x8 b) {
    return _mm256_cmp_ps(a, b, _CMP_NGE_UQ);
}

struct ReduceArgMax {
    f32x8 v = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    f32x8 idx = _mm256_setzero_ps();
    void step(f32x8 input, size_t index) {
        auto is_nan = isNAN(v, input);
        auto cmp = LessThan(v, input);
        v = Select(cmp, input, v);
        v = Select(is_nan, _mm256_set1_ps(NAN), v);
        idx = Select(cmp, _mm256_set1_ps(float(index)), idx);
        idx = Select(is_nan, _mm256_set1_ps(NAN), idx);
    }
    operator f32x8() { return idx; }
};

struct ReduceRank {
    kun_simd::vec_f32x8 v;
    kun_simd::vec_f32x8 less_count = 0;
    kun_simd::vec_f32x8 eq_count = 0;
    ReduceRank(f32x8 cur) : v{cur} {}
    void step(f32x8 input, size_t index) {
        using namespace kun_simd;
        auto is_nan = sc_isnan(v, input);
        auto cmpless = input < v;
        auto cmpeq = input == v;
        less_count = sc_select(is_nan, NAN, less_count);
        less_count = sc_select(cmpless, less_count + 1.0f, less_count);
        eq_count = sc_select(cmpeq, eq_count + 1.0f, eq_count);
    }
    operator f32x8() { return less_count + (eq_count + 1.0f) / 2.0f; }
};

inline f32x8 Abs(f32x8 a) { return kun_simd::sc_abs(kun_simd::vec_f32x8(a)); }

inline f32x8 Add(f32x8 a, f32x8 b) { return _mm256_add_ps(a, b); }
inline f32x8 Add(f32x8 a, float b) {
    return _mm256_add_ps(a, _mm256_set1_ps(b));
}
inline f32x8 Sub(f32x8 a, f32x8 b) { return _mm256_sub_ps(a, b); }
inline f32x8 Sub(float a, f32x8 b) {
    return _mm256_sub_ps(_mm256_set1_ps(a), b);
}
inline f32x8 Sub(f32x8 a, float b) {
    return _mm256_sub_ps(a, _mm256_set1_ps(b));
}
inline f32x8 Mul(f32x8 a, f32x8 b) { return _mm256_mul_ps(a, b); }
inline f32x8 Mul(f32x8 a, float b) {
    return _mm256_mul_ps(a, _mm256_set1_ps(b));
}
inline f32x8 Div(f32x8 a, f32x8 b) { return _mm256_div_ps(a, b); }
inline f32x8 Div(f32x8 a, float b) {
    return _mm256_div_ps(a, _mm256_set1_ps(b));
}

inline f32x8 Or(f32x8 a, f32x8 b) { return _mm256_or_ps(a, b); }
inline f32x8 And(f32x8 a, f32x8 b) { return _mm256_and_ps(a, b); }
inline f32x8 Not(f32x8 a) {
    return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
}

inline f32x8 Sqrt(f32x8 a) { return _mm256_sqrt_ps(a); }

inline f32x8 constVec(float v) { return _mm256_set1_ps(v); }

inline f32x8 Sign(f32x8 in) {
    using namespace kun_simd;
    vec_f32x8 v = in;
    auto is_nan = sc_isnan(v);
    auto v1 = sc_select(is_nan, NAN, 1.0f);
    v1 = sc_select(v < 0.0f, -1.0f, v1);
    v1 = sc_select(v == 0.0f, 0.0f, v1);
    return v1;
}

inline f32x8 Log(f32x8 a) {
    alignas(32) float v[8];
    _mm256_store_ps(v, a);
    for (int i = 0; i < 8; i++) {
        v[i] = std::log(v[i]);
    }
    return _mm256_load_ps(v);
}

inline f32x8 SetInfOrNanToZero(f32x8 a) {
    auto mask = isNAN(_mm256_sub_ps(a, a));
    return Select(mask, _mm256_setzero_ps(), a);
}

} // namespace ops
} // namespace kun
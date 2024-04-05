#pragma once

#include "Base.hpp"
#include "Math.hpp"
#include "StreamBuffer.hpp"
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

template <typename T1, typename T2, typename T3>
inline T1 Select(T1 cond, T2 vtrue, T3 vfalse) {
    return kun_simd::sc_select(cond, vtrue, vfalse);
}

template <typename T, int stride>
struct InputSTs : DataSource<true> {
    using simd_t = kun_simd::vec<T, stride>;
    T *__restrict buf;
    InputSTs(T *base, size_t stock_idx, size_t num_stock, size_t total_time,
             size_t start)
        : buf{base + stock_idx * total_time * stride + start * stride} {}
    simd_t step(size_t index) { return simd_t::load(&buf[index * stride]); }

    simd_t getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return NAN;
        }
        return simd_t::load(&buf[(index - offset) * stride]);
    }

    simd_t getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

template <typename T, int stride>
struct InputTS : DataSource<true> {
    using simd_t = kun_simd::vec<T, stride>;
    T *__restrict buf;
    size_t stock_idx;
    size_t num_stock;
    InputTS(T *base, size_t stock_idx, size_t num_stock, size_t length,
            size_t start)
        : buf{base + start * num_stock}, stock_idx{stock_idx},
          num_stock{num_stock} {}

    T *getPtr(size_t index) {
        return &buf[index * num_stock + stride * stock_idx];
    }

    simd_t step(size_t index) { return simd_t::load(getPtr(index)); }

    simd_t getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return NAN;
        }
        return simd_t::load(getPtr(index - offset));
    }
    simd_t getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

template <typename T, int stride>
struct OutputSTs : DataSource<true> {
    using simd_t = kun_simd::vec<T, stride>;
    T *__restrict buf;
    OutputSTs(T *base, size_t stock_idx, size_t num_stock, size_t length,
              size_t start)
        : buf{base + stock_idx * length * stride} {}
    void store(size_t index, const simd_t &v) {
        simd_t::store(v, &buf[index * stride]);
    }

    simd_t getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return NAN;
        }
        return simd_t::load(&buf[(index - offset) * stride]);
    }
    simd_t getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

template <typename T, int stride>
struct OutputTS : DataSource<true> {
    using simd_t = kun_simd::vec<T, stride>;
    T *__restrict buf;
    size_t stock_idx;
    size_t num_stock;
    OutputTS(T *base, size_t stock_idx, size_t num_stock, size_t length,
             size_t start)
        : buf{base}, stock_idx{stock_idx}, num_stock{num_stock} {}

    T *getPtr(size_t index) {
        return &buf[index * num_stock + stride * stock_idx];
    }
    void store(size_t index, const simd_t &v) {
        simd_t::store(v, getPtr(index));
    }

    simd_t getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return NAN;
        }
        return simd_t::load(getPtr(index - offset));
    }
    simd_t getWindowUnordered(size_t index, size_t offset) {
        return getWindow(index, offset);
    }
};

template <typename T, int stride, size_t window>
struct OutputWindow : DataSource<true> {
    using simd_t = kun_simd::vec<T, stride>;
    // window slots of floatx8
    alignas(alignof(simd_t)) T buf[window * stride];
    // next writable position
    size_t pos;
    OutputWindow() : pos{0} {
        for (size_t i = 0; i < window * stride; i++) {
            buf[i] = NAN;
        }
    }
    void store(size_t index, const simd_t &in) {
        simd_t::store(in, &buf[pos * stride]);
        pos += 1;
        pos = (pos >= window) ? 0 : pos;
    }
    simd_t getWindow(size_t index, size_t offset) {
        offset += 1;
        auto idx = pos >= offset ? (pos - offset) : (pos + window - offset);
        return simd_t::load(&buf[idx * stride]);
    }
    simd_t getWindowUnordered(size_t index, size_t offset) {
        return simd_t::load(&buf[offset * stride]);
    }
};

template <typename T, int stride, size_t window>
struct StreamWindow : DataSource<true> {
    using simd_t = kun_simd::vec<T, stride>;
    // next writable position
    size_t &pos;
    // window size
    size_t stock_idx;
    size_t num_stock;
    // window slots of floatx8
    T *buf;
    StreamWindow(StreamBuffer<T> *buf, size_t stock_idx, size_t num_stock)
        : pos{*buf->getPos(stock_idx, num_stock, window)}, stock_idx{stock_idx},
          num_stock{num_stock}, buf{buf->getBuffer()} {}
    void store(size_t index, const simd_t &in) {
        simd_t::store(in, &buf[pos * num_stock + stock_idx * stride]);
        pos += 1;
        pos = (pos >= window) ? 0 : pos;
    }
    T *getWindowPtr(size_t index, size_t offset) {
        offset += 1;
        auto idx = pos >= offset ? (pos - offset) : (pos + window - offset);
        return &buf[idx * num_stock + stock_idx * stride];
    }
    simd_t getWindow(size_t index, size_t offset) {
        return simd_t::load(getWindowPtr(index, offset));
    }
    simd_t step(size_t index) { return getWindow(index, 0); }
    simd_t getWindowUnordered(size_t index, size_t offset) {
        return simd_t::load(&buf[offset * stride]);
    }
};

template <typename T, int stride>
struct StreamWindow<T, stride, 1ul> : DataSource<true> {
    using simd_t = kun_simd::vec<T, stride>;
    // window size
    size_t stock_idx;
    // window slots of floatx8
    T *buf;
    StreamWindow(StreamBuffer<T> *buf, size_t stock_idx, size_t num_stock)
        : stock_idx{stock_idx}, buf{buf->getBuffer()} {}
    void store(size_t index, const simd_t &in) {
        simd_t::store(in, &buf[stock_idx * stride]);
    }
    // simd_t getWindow(size_t index, size_t offset) {
    //     return _mm256_load_ps(buf);
    // }
    simd_t step(size_t index) { return simd_t::load(&buf[stock_idx * stride]); }
    // simd_t getWindowUnordered(size_t index, size_t offset) {
    //     return _mm256_load_ps(&buf[offset * stride]);
    // }
};

template <typename TInput>
struct RequireWindow {
    static_assert(TInput::containsWindow, "This stage needs window data");
};

using kun_simd::sc_isnan;
using kun_simd::sc_select;

template <typename T, int stride, int window>
struct FastWindowedSum {
    using simd_t = kun_simd::vec<T, stride>;
    using simd_int_t =
        kun_simd::vec<typename kun_simd::int_type_of_same_size<T>::result, stride>;
    simd_t v = 0;
    simd_int_t num_nans = window;
    template <typename TInput>
    simd_t step(TInput &input, simd_t cur, size_t index) {
        RequireWindow<TInput>{};
        auto old = input.getWindow(index, window);
        auto old_is_nan = sc_isnan(old);
        auto new_is_nan = sc_isnan(cur);
        // v = old_is_nan? v : (v-old)
        v = sc_select(old_is_nan, v, v - old);
        // v = new_is_nan? v : (v+cur)
        v = sc_select(new_is_nan, v, v + cur);
        num_nans = num_nans -
                   sc_select(kun_simd::bitcast<simd_int_t>(old_is_nan), 1, 0);
        num_nans = num_nans +
                   sc_select(kun_simd::bitcast<simd_int_t>(new_is_nan), 1, 0);
        auto out_is_normal = kun_simd::bitcast<simd_t>(num_nans == 0);
        return sc_select(out_is_normal, v, NAN);
    }
};

template <typename T, int stride>
struct ReduceAdd {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = 0;
    void step(simd_t input, size_t index) { v = v + input; }
    operator simd_t() { return v; }
};

template <typename T, int stride>
struct ReduceMul {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = T(1.0);
    void step(simd_t input, size_t index) { v = v * input; }
    operator simd_t() { return v; }
};

template <typename T, int stride>
struct ReduceMin {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = std::numeric_limits<T>::infinity();
    void step(simd_t input, size_t index) { v = sc_min(v, input); }
    operator simd_t() { return v; }
};

template <typename T, int stride>
struct ReduceMax {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = -std::numeric_limits<T>::infinity();
    void step(simd_t input, size_t index) { v = sc_max(v, input); }
    operator simd_t() { return v; }
};

template <typename T, int stride, int window>
struct ReduceDecayLinear {
    using simd_t = kun_simd::vec<T, stride>;
    static constexpr T stepSize() {
        return 1.0 / ((1.0 + window) * window / 2);
    }
    simd_t weight = window * stepSize();
    simd_t v = 0;
    void step(simd_t input, size_t index) {
        v = sc_fmadd(input, weight, v);
        weight = weight - stepSize();
    }
    operator simd_t() { return v; }
};

template <typename T, int stride, int window, typename TInput>
kun_simd::vec<T, stride> windowedRef(TInput &input, size_t index) {
    RequireWindow<TInput>{};
    if (index >= window) {
        return input.getWindow(index, window);
    }
    return NAN;
}

template <typename T, int stride, int window, typename TInput>
kun_simd::vec<T, stride> windowedRefStream(TInput &input, size_t index) {
    RequireWindow<TInput>{};
    return input.getWindow(index, window);
}

template <typename T1, typename T2>
inline auto LessThan(T1 a, T2 b) -> decltype(kun_simd::operator<(a, b)) {
    return kun_simd::operator<(a, b);
}

template <typename T1, typename T2>
inline auto LessEqual(T1 a, T2 b) -> decltype(kun_simd::operator<=(a, b)) {
    return kun_simd::operator<=(a, b);
}

template <typename T1, typename T2>
inline auto GreaterThan(T1 a, T2 b) -> decltype(kun_simd::operator>(a, b)) {
    return kun_simd::operator>(a, b);
}

template <typename T1, typename T2>
inline auto GreaterEqual(T1 a, T2 b) -> decltype(kun_simd::operator>=(a, b)) {
    return kun_simd::operator>=(a, b);
}

template <typename T1, typename T2>
inline auto Equals(T1 a, T2 b) -> decltype(kun_simd::operator==(a, b)) {
    return kun_simd::operator==(a, b);
}

// inline f32x8 LessThanOrNan(f32x8 a, f32x8 b) {
//     return _mm256_cmp_ps(a, b, _CMP_NGE_UQ);
// }

template <typename T, int stride>
struct ReduceArgMax {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = -std::numeric_limits<T>::infinity();
    simd_t idx = 0;
    void step(simd_t input, size_t index) {
        auto is_nan = sc_isnan(v, input);
        auto cmp = v < input;
        v = sc_select(cmp, input, v);
        v = sc_select(is_nan, NAN, v);
        idx = sc_select(cmp, T(index), idx);
        idx = sc_select(is_nan, NAN, idx);
    }
    operator simd_t() { return idx; }
};

template <typename T, int stride>
struct ReduceArgMin {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = std::numeric_limits<T>::infinity();
    simd_t idx = 0;
    void step(simd_t input, size_t index) {
        auto is_nan = sc_isnan(v, input);
        auto cmp = GreaterThan(v, input);
        v = sc_select(cmp, input, v);
        v = sc_select(is_nan, NAN, v);
        idx = sc_select(cmp, T(index), idx);
        idx = sc_select(is_nan, NAN, idx);
    }
    operator simd_t() { return idx; }
};

template <typename T, int stride>
struct ReduceRank {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v;
    simd_t less_count = 0;
    simd_t eq_count = 0;
    ReduceRank(simd_t cur) : v{cur} {}
    void step(simd_t input, size_t index) {
        using namespace kun_simd;
        simd_t input2 = input;
        auto is_nan = sc_isnan(v, input2);
        auto cmpless = input2 < v;
        auto cmpeq = input2 == v;
        less_count = sc_select(is_nan, NAN, less_count);
        less_count = sc_select(cmpless, less_count + T(1.0), less_count);
        eq_count = sc_select(cmpeq, eq_count + T(1.0), eq_count);
    }
    operator simd_t() { return less_count + (eq_count + T(1.0)) / T(2.0); }
};

template <typename T1, typename T2>
inline auto Max(T1 a, T2 b) -> decltype(kun_simd::sc_max(a, b)) {
    return kun_simd::sc_max(a, b);
}

template <typename T1, typename T2>
inline auto Min(T1 a, T2 b) -> decltype(kun_simd::sc_min(a, b)) {
    return kun_simd::sc_min(a, b);
}

template <typename T1>
struct DecayVec {
    using result = decltype(kun_simd::sc_abs(std::declval<T1>()));
};

template <typename T1>
using DecayVec_t = typename DecayVec<T1>::result;

template <typename T1>
inline DecayVec_t<T1> Abs(T1 a) {
    return kun_simd::sc_abs(a);
}

template <typename T1, typename T2>
inline auto Add(T1 a, T2 b) -> decltype(kun_simd::operator+(a, b)) {
    return kun_simd::operator+(a, b);
}

template <typename T1, typename T2>
inline auto Sub(T1 a, T2 b) -> decltype(kun_simd::operator-(a, b)) {
    return kun_simd::operator-(a, b);
}

template <typename T1, typename T2>
inline auto Mul(T1 a, T2 b) -> decltype(kun_simd::operator*(a, b)) {
    return kun_simd::operator*(a, b);
}

template <typename T1, typename T2>
inline auto Div(T1 a, T2 b) -> decltype(kun_simd::operator/(a, b)) {
    return kun_simd::operator/(a, b);
}

template <typename T1, typename T2>
inline auto Or(T1 a, T2 b) -> decltype(kun_simd::operator|(a, b)) {
    return kun_simd::operator|(a, b);
}

template <typename T1, typename T2>
inline auto And(T1 a, T2 b) -> decltype(kun_simd::operator&(a, b)) {
    return kun_simd::operator&(a, b);
}

template <typename T1>
inline DecayVec_t<T1> Not(T1 a) {
    return kun_simd::operator!(a);
}

template <typename T1>
inline DecayVec_t<T1> Sqrt(T1 a) {
    return kun_simd::sc_sqrt(a);
}

template <int lanes, typename T>
inline kun_simd::vec<T, lanes> constVec(const T &v) {
    return kun_simd::vec<T, lanes>{v};
}

template <typename T1>
inline DecayVec_t<T1> Sign(T1 v) {
    using namespace kun_simd;
    auto is_nan = sc_isnan(v);
    auto v1 = sc_select(is_nan, NAN, 1.0f);
    v1 = sc_select(v < 0.0f, -1.0f, v1);
    v1 = sc_select(v == 0.0f, 0.0f, v1);
    return v1;
}

template <typename TIn>
inline DecayVec_t<TIn> Log(TIn a0) {
    DecayVec_t<TIn> a = a0;
    using T = typename DecayVec_t<TIn>::T;
    constexpr int lanes = DecayVec_t<TIn>::lanes;
    T *v = a.raw;
    for (int i = 0; i < lanes; i++) {
        v[i] = std::log(v[i]);
    }
    return a;
}

template <typename T, typename T2>
inline DecayVec_t<T> SetInfOrNanToValue(T aa, T2 v) {
    DecayVec_t<T> a = aa;
    auto mask = sc_isnan(a - a);
    return sc_select(mask, v, a);
}

} // namespace ops
} // namespace kun
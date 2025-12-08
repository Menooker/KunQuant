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
inline DecayVec_t<T2> Select(T1 cond, T2 vtrue, T3 vfalse) {
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

    T getWindowLane(size_t index, size_t offset, size_t lane) {
        if (index < offset) {
            return NAN;
        }
        return buf[(index - offset) * stride + lane];
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
    simd_t step(size_t index, const typename simd_t::Masktype &mask) {
        return simd_t::masked_load(getPtr(index), mask);
    }

    simd_t getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return NAN;
        }
        return simd_t::load(getPtr(index - offset));
    }

    T getWindowLane(size_t index, size_t offset, size_t lane) {
        if (index < offset) {
            return NAN;
        }
        return getPtr(index - offset)[lane];
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

    T getWindowLane(size_t index, size_t offset, size_t lane) {
        if (index < offset) {
            return NAN;
        }
        return buf[(index - offset) * stride + lane];
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
    void store(size_t index, const simd_t &v,
               const typename simd_t::Masktype &mask) {
        simd_t::masked_store(v, getPtr(index), mask);
    }

    simd_t getWindow(size_t index, size_t offset) {
        if (index < offset) {
            return NAN;
        }
        return simd_t::load(getPtr(index - offset));
    }

    T getWindowLane(size_t index, size_t offset, size_t lane) {
        if (index < offset) {
            return NAN;
        }
        return getPtr(index - offset)[lane];
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

    T *getPtr(size_t offset) {
        offset += 1;
        auto idx = pos >= offset ? (pos - offset) : (pos + window - offset);
        return &buf[idx * stride];
    }

    simd_t getWindow(size_t index, size_t offset) {
        return simd_t::load(getPtr(offset));
    }

    T getWindowLane(size_t index, size_t offset, size_t lane) {
        return getPtr(offset)[lane];
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
        : pos{*buf->getPos(stock_idx, roundUp(num_stock, stride), window)},
          stock_idx{stock_idx}, num_stock{roundUp(num_stock, stride)},
          buf{buf->getBuffer()} {}
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

    T getWindowLane(size_t index, size_t offset, size_t lane) {
        return getWindowPtr(index, offset)[lane];
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

template <typename T, int stride>
INLINE kun_simd::vec<T, stride>
kahanAdd(typename kun_simd::vec<T, stride>::Masktype isnan_small,
         kun_simd::vec<T, stride> sum, kun_simd::vec<T, stride> small,
         kun_simd::vec<T, stride> &compensation) {
    auto y = small - compensation;
    auto t = sum + y;
    compensation = sc_select(isnan_small, compensation, t - sum - y);
    return t;
}

template <typename T, int stride>
INLINE kun_simd::vec<T, stride>
kahanAdd(kun_simd::vec<T, stride> sum, kun_simd::vec<T, stride> small,
         kun_simd::vec<T, stride> &compensation) {
    auto y = small - compensation;
    auto t = sum + y;
    compensation = t - sum - y;
    return t;
}

template <typename T, int stride>
struct Accumulator {
    using simd_t = kun_simd::vec<T, stride>;
    using float_mask_t = typename simd_t::Masktype;
    simd_t v = 0;
    struct Value {
        simd_t v;
        Accumulator& acc;
        operator simd_t() const { return v; }
    };
    Value asValue() {
        return Value{v, *this};
    }
};

template <typename T, int stride>
kun_simd::vec<T, stride>
SetAccumulator(const typename Accumulator<T, stride>::Value& accu,
               const typename Accumulator<T, stride>::float_mask_t &mask,
               const kun_simd::vec<T, stride> &v) {
    accu.acc.v = sc_select(mask, v, accu.v);
    return accu.acc.v;
}

template <typename T, int stride, int window>
struct FastWindowedSum {
    using simd_t = kun_simd::vec<T, stride>;
    using simd_int_t =
        kun_simd::vec<typename kun_simd::fp_trait<T>::int_t, stride>;
    using int_mask_t = typename simd_int_t::Masktype;
    using float_mask_t = typename simd_t::Masktype;
    simd_t v = 0;
    simd_t compensationAdd = 0;
    simd_t compensationSub = 0;
    simd_int_t num_nans = window;
    template <typename TInput>
    simd_t step(TInput &input, simd_t cur, size_t index) {
        RequireWindow<TInput>{};
        auto old = input.getWindow(index, window);
        auto old_is_nan = sc_isnan(old);
        auto new_is_nan = sc_isnan(cur);
        // v = old_is_nan? v : (v-old)
        v = sc_select(old_is_nan, v,
                      kahanAdd(old_is_nan, v, 0 - old, compensationSub));
        // v = new_is_nan? v : (v+cur)
        v = sc_select(new_is_nan, v,
                      kahanAdd(new_is_nan, v, cur, compensationAdd));
        num_nans =
            num_nans - sc_select(kun_simd::bitcast<int_mask_t>(old_is_nan),
                                 simd_int_t{1}, simd_int_t{0});
        num_nans =
            num_nans + sc_select(kun_simd::bitcast<int_mask_t>(new_is_nan),
                                      simd_int_t{1}, simd_int_t{0});
        auto out_is_normal = kun_simd::bitcast<float_mask_t>(num_nans == 0);
        return sc_select(out_is_normal, v, simd_t{NAN});
    }
};

template <typename T, int stride, int window>
struct ExpMovingAvg {
    using simd_t = kun_simd::vec<T, stride>;
    using simd_int_t =
        kun_simd::vec<typename kun_simd::fp_trait<T>::int_t, stride>;
    simd_t v;
    ExpMovingAvg(const simd_t &init) : v{init} {}
    static constexpr T weight_latest = T(2.0) / (window + 1);
    simd_t step(simd_t cur, size_t index) {
        auto is_nan = sc_isnan(cur);
        auto old_is_nan = sc_isnan(v);
        auto newv = v * (1 - weight_latest) + cur * weight_latest;
        v = sc_select(is_nan, v, newv);
        v = sc_select(old_is_nan, cur, v);
        return v;
    }
};

template <typename T, int stride, int window>
struct WindowedLinearRegression {
    using simd_t = kun_simd::vec<T, stride>;
    using simd_int_t =
        kun_simd::vec<typename kun_simd::fp_trait<T>::int_t, stride>;
    simd_t i_sum = T(0);
    simd_t x_sum = T(0);
    simd_t x2_sum = T(0);
    simd_t y_sum = T(0);
    simd_t y2_sum = T(0);
    simd_t xy_sum = T(0);
    simd_int_t num_nans = window;
    using int_mask_t = typename simd_int_t::Masktype;
    using float_mask_t = typename simd_t::Masktype;

    template <typename TInput>
    const WindowedLinearRegression &step(TInput &input, simd_t cur,
                                         size_t index) {
        // implementation from
        // https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/_libs/rolling.pyx#L137
        RequireWindow<TInput>{};
        xy_sum = xy_sum - y_sum;
        x2_sum = x2_sum + i_sum - T(2.0) * x_sum;
        x_sum = x_sum - i_sum;
        auto old = input.getWindow(index, window);
        auto old_is_nan = sc_isnan(old);
        auto new_is_nan = sc_isnan(cur);
        i_sum = sc_select(old_is_nan, i_sum, i_sum - T(1));
        y_sum = sc_select(old_is_nan, y_sum, y_sum - old);
        y2_sum = sc_select(old_is_nan, y2_sum, y2_sum - old * old);
        num_nans =
            num_nans - sc_select(kun_simd::bitcast<int_mask_t>(old_is_nan),
                                 simd_int_t{1}, simd_int_t{0});
        num_nans =
            num_nans + sc_select(kun_simd::bitcast<int_mask_t>(new_is_nan),
                                 simd_int_t{1}, simd_int_t{0});

        i_sum = sc_select(new_is_nan, i_sum, i_sum + T(1));
        x_sum = sc_select(new_is_nan, x_sum, x_sum + T(window));
        x2_sum =
            sc_select(new_is_nan, x2_sum, x2_sum + (T(window) * T(window)));
        y_sum = sc_select(new_is_nan, y_sum, y_sum + cur);
        y2_sum = sc_select(new_is_nan, y2_sum, y2_sum + cur * cur);
        xy_sum = sc_select(new_is_nan, xy_sum, xy_sum + T(window) * cur);
        return *this;
    }
};

template <typename T, int stride, int window>
kun_simd::vec<T, stride> WindowedLinearRegressionRSqaureImpl(
    const WindowedLinearRegression<T, stride, window> &v) {
    auto N = kun_simd::fast_cast<kun_simd::vec<T, stride>>(window - v.num_nans);
    auto R1 = (N * v.xy_sum - v.x_sum * v.y_sum);
    R1 = R1 * R1;
    auto R2 =
        (N * v.x2_sum - v.x_sum * v.x_sum) * (N * v.y2_sum - v.y_sum * v.y_sum);
    return R1 / R2;
}

template <typename T, int stride, int window>
kun_simd::vec<T, stride> WindowedLinearRegressionSlopeImpl(
    const WindowedLinearRegression<T, stride, window> &v) {
    auto N = kun_simd::fast_cast<kun_simd::vec<T, stride>>(window - v.num_nans);
    auto slope =
        (N * v.xy_sum - v.x_sum * v.y_sum) / (N * v.x2_sum - v.x_sum * v.x_sum);
    return slope;
}

template <typename T, int stride, int window>
kun_simd::vec<T, stride> WindowedLinearRegressionResiImpl(
    const WindowedLinearRegression<T, stride, window> &v,
    kun_simd::vec<T, stride> val) {
    auto N = kun_simd::fast_cast<kun_simd::vec<T, stride>>(window - v.num_nans);
    auto slope = WindowedLinearRegressionSlopeImpl(v);
    auto x_mean = v.x_sum / N;
    auto y_mean = v.y_sum / N;
    auto interp = y_mean - slope * x_mean;
    return val - (slope * T(window) + interp);
}

template <typename T, int stride, int window, typename T2>
kun_simd::vec<T, stride> WindowedLinearRegressionResiImpl(
    const WindowedLinearRegression<T, stride, window> &v, T2 val) {
    auto N = kun_simd::fast_cast<kun_simd::vec<T, stride>>(window - v.num_nans);
    auto slope = WindowedLinearRegressionSlopeImpl(v);
    auto x_mean = v.x_sum / N;
    auto y_mean = v.y_sum / N;
    auto interp = y_mean - slope * x_mean;
    return kun_simd::vec<T, stride>(val) - (slope * T(window) + interp);
}

template <typename T, int stride>
struct ReduceAdd {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = 0;
    simd_t compensation = 0;
    void step(simd_t input, size_t index) {
        v = kahanAdd(v, input, compensation);
    }
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
    void step(simd_t input, size_t index) {
        v = sc_select(sc_isnan(v, input), NAN, sc_min(v, input));
    }
    operator simd_t() { return v; }
};

template <typename T, int stride>
struct ReduceMax {
    using simd_t = kun_simd::vec<T, stride>;
    simd_t v = -std::numeric_limits<T>::infinity();
    void step(simd_t input, size_t index) {
        v = sc_select(sc_isnan(v, input), NAN, sc_max(v, input));
    }
    operator simd_t() { return v; }
};

template <typename T, int stride, int window>
struct ReduceDecayLinear {
    using simd_t = kun_simd::vec<T, stride>;
    static constexpr T stepSize() {
        return 1.0 / ((1.0 + window) * window / 2);
    }
    simd_t weight = stepSize();
    simd_t v = 0;
    void step(simd_t input, size_t index) {
        v = sc_fmadd(input, weight, v);
        weight = weight + stepSize();
    }
    operator simd_t() { return v; }
};

template <typename T, int stride, int window, size_t streamwindow>
kun_simd::vec<T, stride>
windowedRef(StreamWindow<T, stride, streamwindow> &input, size_t index) {
    return input.getWindow(index, window);
}

template <typename T, int stride, int window, typename TInput>
kun_simd::vec<T, stride> windowedRef(TInput &input, size_t index) {
    RequireWindow<TInput>{};
    if (index >= window) {
        return input.getWindow(index, window);
    }
    return NAN;
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
        auto cmp = v > input;
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

#ifdef __AVX512F__
inline __mmask8 Or(__mmask8 a, __mmask8 b) { return a | b; }

inline __mmask8 And(__mmask8 a, __mmask8 b) { return a & b; }

inline __mmask8 Not(__mmask8 a) { return ~a; }

inline __mmask16 Or(__mmask16 a, __mmask16 b) { return a | b; }

inline __mmask16 And(__mmask16 a, __mmask16 b) { return a & b; }

inline __mmask16 Not(__mmask16 a) { return ~a; }
#endif

template <typename T1, typename T2>
inline auto Or(T1 a, T2 b) -> decltype(kun_simd::operator|(a, b)) {
    return kun_simd::operator|(a, b);
}

template <typename T1, typename T2>
inline auto And(T1 a, T2 b) -> decltype(kun_simd::operator&(a, b)) {
    return kun_simd::operator&(a, b);
}

template <typename T1>
inline auto Not(T1 a) -> decltype(kun_simd::operator!(a)) {
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
    using simd_t = DecayVec_t<T1>;
    auto is_nan = sc_isnan(v);
    auto v1 = sc_select(is_nan, simd_t{NAN}, simd_t{1.0f});
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
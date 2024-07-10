#pragma once

#include <KunSIMD/Vector.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>

namespace kun {
namespace ops {

namespace quantile {
template <typename T>
T call(T *sorted, int length, T q) {
    T findex = (length - 1) * q;
    int index = findex;
    T fraction = findex - index;
    if (std::abs(fraction) < 1e-6) {
        return sorted[index];
    }
    T i = sorted[index];
    T j = sorted[index + 1];
    return i + (j - i) * fraction;
}

} // namespace quantile
template <typename T, int stride, int window, typename TInput>
kun_simd::vec<T, stride> windowedQuantile(TInput &input, size_t index, T q) {
    alignas(alignof(kun_simd::vec<T, stride>)) T result[stride];
    T sorted[window];
    for (int i = 0; i < stride; i++) {
        int cnt = 0;
        for (int j = 0; j < window; j++) {
            auto v = input.getWindowLane(index, j, i);
            if (!std::isnan(v)) {
                sorted[cnt] = v;
                cnt++;
            }
        }
        if (cnt > 0) {
            std::sort(sorted, sorted + cnt);
            result[i] = quantile::call<T>(sorted, cnt, q);
        } else {
            result[i] = NAN;
        }
    }
    return kun_simd::vec<T, stride>::load(result);
}
} // namespace ops
} // namespace kun
#pragma once

#include <Kun/SkipList.hpp>
#include <KunSIMD/Vector.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>

namespace kun {
namespace ops {

namespace {
template <typename T, int simdLen>
struct SkipListStateImpl {
    using simd_t = kun_simd::vec<T, simdLen>;
    SkipList skipList[simdLen];
    int lastInsertRank[simdLen];
    size_t window;
    SkipListStateImpl(size_t window) : window(window) {
        for (int i = 0; i < simdLen; i++) {
            skipList[i].init(window);
        }
    }
    SkipListStateImpl &step(const simd_t &oldvalue, const simd_t &value,
                            size_t index) {
        alignas(alignof(simd_t)) T values[simdLen];
        simd_t::store(oldvalue, values);
        for (int i = 0; i < simdLen; i++) {
            if (!std::isnan(values[i])) {
                skipList[i].remove(values[i]);
            }
        }
        simd_t::store(value, values);
        for (int i = 0; i < simdLen; i++) {
            if (!std::isnan(values[i])) {
                lastInsertRank[i] = skipList[i].insert(values[i], index);
            }
        }
        return *this;
    }
};

template <typename T, int simdLen, int expectedwindow>
struct SkipListState : SkipListStateImpl<T, simdLen> {
    SkipListState() : SkipListStateImpl<T, simdLen>(expectedwindow) {}
};
} // namespace


template <typename T, int simdLen, int expectedwindow>
struct Serializer<SkipListState<T, simdLen, expectedwindow>> {
    static bool serialize(StateBuffer *obj, OutputStreamBase *stream) {
        for(size_t i = 0; i < obj->num_objs; i++) {
            auto &state = obj->get<SkipListState<T, simdLen, expectedwindow>>(i);
            if (!serializeSkipList(state.skipList, state.lastInsertRank, simdLen,
                                   expectedwindow, stream)) {
                return false;
            }
        }
        return true;
    }

    static bool deserialize(StateBuffer *obj, InputStreamBase *stream) {
        for(size_t i = 0; i < obj->num_objs; i++) {
            auto &state = obj->get<SkipListState<T, simdLen, expectedwindow>>(i);
            new (&state) SkipListState<T, simdLen, expectedwindow>{};
            if (!deserializeSkipList(state.skipList, state.lastInsertRank, simdLen,
                                     expectedwindow, stream)) {
                return false;
            }
        }
        return true;
    }
};

// https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx

template <typename T, int stride>
kun_simd::vec<T, stride> SkipListQuantile(SkipListStateImpl<T, stride> &state,
                                          T q) {
    alignas(alignof(kun_simd::vec<T, stride>)) T result[stride];
    size_t index;
    bool found;
    for (int i = 0; i < stride; i++) {
        int nobs = state.skipList[i].size();
        if (nobs != state.window) {
            result[i] = NAN;
            continue;
        }
        T idx_with_fraction = q * (nobs - 1);
        int idx = static_cast<int>(idx_with_fraction);
        if (idx == idx_with_fraction) {
            result[i] = state.skipList[i].get(idx, index, found);
        } else {
            auto vlow = state.skipList[i].get(idx, index, found);
            auto vhigh = state.skipList[i].get(idx + 1, index, found);
            result[i] = vlow + (vhigh - vlow) * (idx_with_fraction - idx);
        }
    }
    return kun_simd::vec<T, stride>::load(result);
}

template <typename T, int stride>
kun_simd::vec<T, stride> SkipListRank(SkipListStateImpl<T, stride> &state,
                                      const kun_simd::vec<T, stride> &cur) {
    alignas(alignof(kun_simd::vec<T, stride>)) T result[stride];
    alignas(alignof(kun_simd::vec<T, stride>)) T curval[stride];
    kun_simd::vec<T, stride>::store(cur, curval);
    size_t index;
    bool found;
    for (int i = 0; i < stride; i++) {
        int nobs = state.skipList[i].size();
        if (nobs != state.window) {
            result[i] = NAN;
            continue;
        }
        double rank = state.lastInsertRank[i] + 1;
        double rank_min = state.skipList[i].minRank(curval[i]) + 1;
        rank = (((rank * (rank + 1) / 2) - ((rank_min - 1) * rank_min / 2)) /
                (rank - rank_min + 1));
        result[i] = rank;
    }
    return kun_simd::vec<T, stride>::load(result);
}

template <typename T, int stride>
kun_simd::vec<T, stride> SkipListMinMax(SkipListStateImpl<T, stride> &state, bool is_min) {
    alignas(alignof(kun_simd::vec<T, stride>)) T result[stride];
    size_t index;
    bool found;
    for (int i = 0; i < stride; i++) {
        int nobs = state.skipList[i].size();
        if (nobs != state.window) {
            result[i] = NAN;
            continue;
        }
        int rank = is_min ? 0 : state.window - 1;
        result[i] = state.skipList[i].get(rank, index, found);
    }
    return kun_simd::vec<T, stride>::load(result);
}

template <typename T, int stride>
kun_simd::vec<T, stride> SkipListMin(SkipListStateImpl<T, stride> &state) {
    return SkipListMinMax(state, true);
}

template <typename T, int stride>
kun_simd::vec<T, stride> SkipListMax(SkipListStateImpl<T, stride> &state) {
    return SkipListMinMax(state, false);
}

template <typename T, int stride>
kun_simd::vec<T, stride> SkipListArgMin(SkipListStateImpl<T, stride> &state, size_t cur_idx) {
    alignas(alignof(kun_simd::vec<T, stride>)) T result[stride];
    size_t index;
    bool found;
    for (int i = 0; i < stride; i++) {
        int nobs = state.skipList[i].size();
        if (nobs != state.window) {
            result[i] = NAN;
            continue;
        }
        state.skipList[i].get(0, index, found);
        result[i] = index + nobs - cur_idx;
    }
    return kun_simd::vec<T, stride>::load(result);
}

} // namespace ops
} // namespace kun
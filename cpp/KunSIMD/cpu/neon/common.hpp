#pragma once

namespace kun_simd {
namespace {
template <typename T, typename U, int N>
void simulate_masked_store(T (&v)[N], U (&mask)[N], T *p) {
    for (int i = 0; i < N; i++) {
        if (mask[i]) {
            p[i] = v[i];
        }
    }
}

template <typename T, typename U, int N>
void simulate_masked_load(T (&v)[N], U (&mask)[N], const T *p) {
    for (int i = 0; i < N; i++) {
        if (mask[i]) {
            v[i] = p[i];
        }
    }
}
} // namespace
} // namespace kun_simd

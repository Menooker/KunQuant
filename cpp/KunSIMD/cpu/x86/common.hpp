#pragma once

#ifdef __AVX2__
#define AVX_IMPL(avx, avx2, a, b) return avx2((a), (b));
// expand to _mm256_op_pspd(a, b)
#define AVX_USE_FP_OP(op, pspd, a, b) return _mm256_##op##_si256((a), (b));
#define AVX_CONST_IMPL(avx, avx2, a, b) return avx2(a, b);
#else
#define AVX_IMPL(avx, avx2, a, b)                                              \
    __m128i a1 = _mm256_castsi256_si128((a));                                  \
    __m128i a2 = _mm256_extractf128_si256((a), 1);                             \
    __m128i b1 = _mm256_castsi256_si128((b));                                  \
    __m128i b2 = _mm256_extractf128_si256((b), 1);                             \
    a1 = avx(a1, b1);                                                          \
    a2 = avx(a2, b2);                                                          \
    return _mm256_setr_m128i(a1, a2);

#define AVX_CONST_IMPL(avx, avx2, a, b)                                        \
    __m128i a1 = _mm256_castsi256_si128((a));                                  \
    __m128i a2 = _mm256_extractf128_si256((a), 1);                             \
    a1 = avx(a1, b);                                                           \
    a2 = avx(a2, b);                                                           \
    return _mm256_setr_m128i(a1, a2);

#define AVX_USE_FP_OP(op, pspd, a, b)                                          \
    return _mm256_cast##pspd##_si256(_mm256_##op##_##pspd(                     \
        _mm256_castsi256_##pspd(a), _mm256_castsi256_##pspd(b)));
#endif
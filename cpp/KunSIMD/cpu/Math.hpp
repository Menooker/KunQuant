#pragma once

#include <Kun/Base.hpp>
#include <KunSIMD/Vector.hpp>
#include <KunSIMD/cpu/cast.hpp>
#include <KunSIMD/cpu/gather.hpp>
#include <limits>
#include <stdint.h>

namespace kun_simd {

KUN_API extern const uint32_t log_const_table1[32];
KUN_API extern const uint32_t log_const_table2[32];

template <typename T, int lanes>
inline vec<T, lanes> log(vec<T, lanes> inval) {
    using Vec = vec<T, lanes>;
    using VecInt = vec<int32_t, lanes>;
    Vec ZERO = 0.0f;
    Vec ln2 = 0.693147181f;
    Vec ONE_f = 1.0f;
    VecInt ONE_i = 1;
    Vec neg_inf_f = -std::numeric_limits<float>::infinity();
    Vec inf_f = std::numeric_limits<float>::infinity();
    Vec qnan_f = std::numeric_limits<float>::quiet_NaN();
    const int approx_bits = 5;
    const int mantissa_bits = 23;

    auto inval_int = bitcast<VecInt>(inval);
    auto aux1_int = (inval_int >> (mantissa_bits - approx_bits)) & 0x0000001f;
    auto aux2_int = aux1_int >> (approx_bits - 1);
    auto aux3_int = (inval_int >> mantissa_bits) + aux2_int;
    auto aux3_f = cast<Vec>(aux3_int);
    aux2_int = (aux2_int ^ 0x0000007f) << 23;
    inval_int = (inval_int & 0x007fffff) | aux2_int;
    auto aux2_f = gather<4>((const float *)log_const_table1, aux1_int);
    aux2_f = aux2_f * bitcast<Vec>(inval_int) - ONE_f;
    Vec poly_f = 0.199984118f;
    poly_f = sc_fmadd(poly_f, aux2_f, -0.250035613f);
    poly_f = sc_fmadd(poly_f, aux2_f, 0.333333343f);
    poly_f = sc_fmadd(poly_f, aux2_f, -0.5f);
    poly_f = sc_fmadd(poly_f, aux2_f, ONE_f) * aux2_f;
    aux2_f = gather<4>((const float *)log_const_table2, aux1_int);
    aux2_f = sc_fmadd(aux3_f, ln2, aux2_f);
    // two sum algorithm
    auto res_hi = poly_f + aux2_f;
    auto res_lo = res_hi - aux2_f;
    res_lo = res_lo - poly_f;
    res_hi = res_hi + res_lo;
    res_hi = sc_select(inval == ZERO, neg_inf_f, res_hi);
    res_hi = sc_select(inval < ZERO, qnan_f, res_hi);
    res_hi = sc_select(inval == inf_f, inf_f, res_hi);
    auto isnan = sc_isnan(inval);
    res_hi = sc_select(isnan, qnan_f, res_hi);
    res_hi = sc_select(inval == ONE_f, ZERO, res_hi);
    return res_hi;
}

template <typename T, int lanes>
inline vec<T, lanes> exp(vec<T, lanes> inval) {
    using Vec = vec<T, lanes>;
    using VecInt = vec<int32_t, lanes>;

    Vec ZERO = 0.0f;
    Vec ln2 = 0.693147181f;
    Vec minus_ln2 = -0.693147181f;
    Vec one_over_ln2 = 1.442695041f;
    Vec half_float = 0.5f;
    Vec ONE_f = 1.0f;
    VecInt ONE_i = 1;
    Vec overflow_x = 88.72283935f;
    Vec underflow_x = -87.33654785f;
    Vec ret_infinity = std::numeric_limits<float>::infinity();

    // to avoid overflow
    auto a_ = inval;

    // e^x = 2^k_int * e^r
    // k_float = floor(x / ln2 + 0.5f)
    auto k_float = sc_floor(sc_fmadd(a_, one_over_ln2, half_float));
    auto k_int = cast<VecInt>(k_float); // k_int = int(k_float)

    // r = a_ - k_float * ln2;
    auto r = sc_fnmadd(k_float, ln2, a_);

    // table[6] = gen_vec_const(elements, 0.142857143f);

    auto Tn = sc_fmadd(r, 0.166666667f, ONE_f);
    // Tn = Tn * (r / i) + 1
    Tn = sc_fmadd(Tn, r * 0.2f, ONE_f);
    Tn = sc_fmadd(Tn, r * 0.25f, ONE_f);
    Tn = sc_fmadd(Tn, r * 0.333333333f, ONE_f);
    Tn = sc_fmadd(Tn, r * 0.5f, ONE_f);
    Tn = sc_fmadd(Tn, r, ONE_f);

    // 2^k_int, shift to exponent bits position
    auto p = k_int << 23;
    auto result = p + bitcast<VecInt>(Tn);
    auto res = bitcast<Vec>(result);
    res = sc_select(inval > overflow_x, ret_infinity, res);
    res = sc_select(inval < underflow_x, ZERO, res);
    return res;
}

} // namespace kun_simd
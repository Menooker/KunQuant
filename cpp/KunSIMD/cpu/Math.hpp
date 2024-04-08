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

    // int representation of the FP value
    auto inval_int = bitcast<VecInt>(inval);
    // the higher order 5 bits of mantissa
    auto mantissa_high5 =
        (inval_int >> (mantissa_bits - approx_bits)) & 0x0000001f;
    // the higher order 1 bit of mantissa
    auto mantissa_high1 = mantissa_high5 >> (approx_bits - 1);
    // prepare the exponential bits for M (it should be 0 or 1 + 127, based on
    // rounding of mantissa_high1)
    auto E_1 = (mantissa_high1 ^ 0x0000007f) << 23;
    // get value of M: expontential be 0/1, and keep the mantissa same of X
    auto M = (inval_int & 0x007fffff) | E_1;
    auto Ri = gather<4>((const float *)log_const_table1, mantissa_high5);
    auto Z = Ri * bitcast<Vec>(M) - ONE_f;
    Vec poly_f = 0.199984118f;
    poly_f = sc_fmadd(poly_f, Z, -0.250035613f);
    poly_f = sc_fmadd(poly_f, Z, 0.333333343f);
    poly_f = sc_fmadd(poly_f, Z, -0.5f);
    poly_f = sc_fmadd(poly_f, Z, ONE_f) * Z;
    // -log(Ri) - (127 * ln2)
    auto minus_log_Ri_minus_127_mul_ln2 =
        gather<4>((const float *)log_const_table2, mantissa_high5);
    // the exponential value + mantissa_high1 for rounding
    auto expo_rounded = (inval_int >> mantissa_bits) + mantissa_high1;
    // Cast the exponential value in FP format. Note that IEEE Float point
    // format stores E+127 instead of E in the data
    auto E_plus_127 = cast<Vec>(expo_rounded);
    // (E+127) * ln2 + -log(Ri) - (127 * ln2) == E * ln2 - log(Ri)
    auto v2 = sc_fmadd(E_plus_127, ln2, minus_log_Ri_minus_127_mul_ln2);
    // two sum algorithm
    auto res_hi = poly_f + v2;
    auto res_lo = res_hi - v2;
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
    using VecInt = vec<typename fp_trait<T>::int_t, lanes>;

    Vec ZERO = T{0.0};
    Vec ln2 = T{0.6931471805599453};
    Vec minus_ln2 = T{-0.6931471805599453};
    Vec one_over_ln2 = T{1.4426950408889634};
    Vec half_float = T{0.5};
    Vec ONE_f = T{1.0};
    VecInt ONE_i = 1;
    Vec overflow_x = T{88.72283935};   // double can handle more than here
    Vec underflow_x = T{-87.33654785}; // double can handle more than here
    Vec ret_infinity = std::numeric_limits<T>::infinity();

    // to avoid overflow
    auto a_ = inval;

    // e^x = 2^k_int * e^r
    // k_float = floor(x / ln2 + 0.5f)
    auto k_float = sc_floor(sc_fmadd(a_, one_over_ln2, half_float));
    auto k_int = cast<VecInt>(k_float); // k_int = int(k_float)

    // r = a_ - k_float * ln2;
    auto r = sc_fnmadd(k_float, ln2, a_);

    // table[6] = gen_vec_const(elements, 0.142857143f);

    auto Tn = sc_fmadd(r, T{0.16666666666666666}, ONE_f);
    // Tn = Tn * (r / i) + 1
    Tn = sc_fmadd(Tn, r * T{0.2}, ONE_f);
    Tn = sc_fmadd(Tn, r * T{0.25}, ONE_f);
    Tn = sc_fmadd(Tn, r * T{0.3333333333333333}, ONE_f);
    Tn = sc_fmadd(Tn, r * T{0.5}, ONE_f);
    Tn = sc_fmadd(Tn, r, ONE_f);

    // 2^k_int, shift to exponent bits position
    auto p = k_int << fp_trait<T>::fraction_bits;
    auto result = p + bitcast<VecInt>(Tn);
    auto res = bitcast<Vec>(result);
    res = sc_select(inval > overflow_x, ret_infinity, res);
    res = sc_select(inval < underflow_x, ZERO, res);
    return res;
}

} // namespace kun_simd
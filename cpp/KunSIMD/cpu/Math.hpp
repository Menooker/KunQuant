#pragma once

#include <Kun/Base.hpp>
#include <KunSIMD/Vector.hpp>
#include <KunSIMD/cpu/Table.hpp>
#include <KunSIMD/cpu/cast.hpp>
#include <KunSIMD/cpu/gather.hpp>
#include <limits>
#include <stdint.h>

namespace kun_simd {

// make n low order bits 1 and other bits 0
template <typename T>
constexpr T makeNTailingOnes(int n) {
    return ((T{1}) << n) - 1;
}

template <typename T, int lanes>
inline vec<T, lanes> log(vec<T, lanes> inval) {
    // Implementation from oneDNN
    // src/cpu/x64/injectors/jit_uni_eltwise_injector.cpp.
    // From J.-M. Muller and others, Handbook of Floating-Point Arithmetic, 2010
    // Here is a brief mathematics to approximate log(x):
    // x = M * 2^E, where M and E can be easily extracted from the binary
    // representation of FP format
    // log(x) = E * log(2) + log(M), where  -log(2)/2 <= log(M) <= log(2)/2;
    // log(M) = log(1 + Z) - log(R_i), where z  = M * R_i - 1, R_i approximates
    //   1 / M, i is index of one of precomputed values;
    // log(1 + X) ~~ polynomial(X), =>
    // if (x is normal)
    //     log(x) ~~ E * log(2) + polynomial(X) - log(R_i),
    // where log(R_i) is table value.
    using Vec = vec<T, lanes>;
    using int_t = typename fp_trait<T>::int_t;
    using VecInt = vec<int_t, lanes>;
    Vec ZERO = T{0.0};
    Vec ln2 = T{0.6931471805599453};
    Vec ONE_f = T{1.0};
    VecInt ONE_i = 1;
    Vec neg_inf_f = -std::numeric_limits<T>::infinity();
    Vec inf_f = std::numeric_limits<T>::infinity();
    Vec qnan_f = std::numeric_limits<T>::quiet_NaN();
    const int approx_bits = 5;
    const int exponent_bits = fp_trait<T>::exponent_bits;
    const int mantissa_bits = fp_trait<T>::fraction_bits;

    // int representation of the FP value
    auto inval_int = bitcast<VecInt>(inval);
    // the higher order 5 bits of mantissa
    auto mantissa_high5 = logical_shr<mantissa_bits - approx_bits>(inval_int) &
                          makeNTailingOnes<int_t>(approx_bits);
    // the highest order 1 bit of mantissa
    auto mantissa_high1 = logical_shr<(approx_bits - 1)>(mantissa_high5);
    // prepare the exponent bits for M. It should be (0 or -1) + 127, based on
    // rounding bit mantissa_high1. If mantissa_high1, the exponent value will
    // be -1. Otherwise, exponent value will be 0.
    // makeNTailingOnes(exponent_bits-1) is for exponent value 0 in FP format
    auto E_1 = logical_shl<mantissa_bits>(
        mantissa_high1 ^ makeNTailingOnes<int_t>(exponent_bits - 1));
    // get value of M: expontential be 0/-1, and keep the mantissa same of X
    auto M = (inval_int & makeNTailingOnes<int_t>(mantissa_bits)) | E_1;
    auto Ri = gather<sizeof(T)>(LogLookupTable<T>::r_table, mantissa_high5);
    auto Z = Ri * bitcast<Vec>(M) - ONE_f;
    // poly_f ~ log(1+Z)
    Vec poly_f = T{1 / 5.0};
    poly_f = sc_fmadd(poly_f, Z, T{-1 / 4.0});
    poly_f = sc_fmadd(poly_f, Z, T{1 / 3.0});
    poly_f = sc_fmadd(poly_f, Z, T{-1 / 2.0});
    poly_f = sc_fmadd(poly_f, Z, ONE_f) * Z;
    // -log(Ri) - (127 * ln2)
    auto minus_log_Ri_minus_127_mul_ln2 =
        gather<sizeof(T)>(LogLookupTable<T>::logr_table, mantissa_high5);
    // the exponential value + mantissa_high1 for rounding
    auto expo_rounded = logical_shr<mantissa_bits>(inval_int) + mantissa_high1;
    // Cast the exponential value in FP format. Note that IEEE single precision
    // Float point format stores E+127 instead of E in the data
    auto E_plus_127 = fast_cast<Vec>(expo_rounded);
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
    auto k_int = fast_cast<VecInt>(k_float); // k_int = int(k_float)

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
    auto p = logical_shl<fp_trait<T>::fraction_bits>(k_int);
    auto result = p + bitcast<VecInt>(Tn);
    auto res = bitcast<Vec>(result);
    res = sc_select(inval > overflow_x, ret_infinity, res);
    res = sc_select(inval < underflow_x, ZERO, res);
    return res;
}

} // namespace kun_simd
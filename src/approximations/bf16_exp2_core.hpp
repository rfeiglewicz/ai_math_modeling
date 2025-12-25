#ifndef BF16_EXP2_CORE_HPP
#define BF16_EXP2_CORE_HPP

#include "../utils/fp_utils.hpp"
#include "../../modeling/coeff_gen/bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"

// Typ stałoprzecinkowy: 1 bit całkowity, 16 ułamkowych.
typedef ac_fixed<17, 1, false> mant_t;

struct PolyResult {
    ac_fixed<41, 1, false> mantissa; // 1 bit całkowity + 40 ułamkowych (pełna precyzja calc_t)
    int32_t exponent;
};

/**
 * @brief Oblicza 2^x przy użyciu aproksymacji odcinkowo-liniowej.
 * Zwraca 40 bitów ułamkowych, aby zapewnić idealne zaokrąglenie RNE.
 */
inline PolyResult bf16_exp2_poly(mant_t mant_val, bool sign) {
    uint8_t lut_index = mant_val.slc<6>(10);

    float a_float = bf16_exp2::coeffs_a[63 - lut_index];
    float b_float = bf16_exp2::coeffs_b[63 - lut_index];

    typedef ac_fixed<32, 8, true> coeffs_t;
    typedef ac_fixed<56, 16, true> calc_t; // calc_t ma 40 bitów ułamkowych (56-16)
    coeffs_t a_fixed = a_float;
    coeffs_t b_fixed = b_float;

    coeffs_t x_fixed = mant_val;
    if (sign) x_fixed = -x_fixed;

    calc_t ax = a_fixed * x_fixed;
    calc_t res = ax + b_fixed;
    
    int msb_idx = -1;
    for (int i = 54; i >= 0; --i) {
        if (res[i]) {
            msb_idx = i;
            break;
        }
    }

    PolyResult result;
    // Waga bitu k w calc_t to 2^(k - 40). Jeśli msb_idx = 40, wykładnik wynosi 0.
    result.exponent = msb_idx - 40;

    // Pobranie MSB i 40 bitów poniżej niego
    ac_int<41, false> raw_bits = 0;
    for (int i = 0; i < 41; ++i) {
        int bit_idx = msb_idx - i;
        if (bit_idx >= 0 && res[bit_idx]) {
            raw_bits[40 - i] = 1;
        }
    }
    result.mantissa.set_slc(0, raw_bits);
    
    return result;
}

inline FPRaw bf16_exp2_core_approx(const FPRaw& input_parts) {
    FPRaw result = {};
    mant_t mant_val;
    int32_t temp_exponent = input_parts.exponent;
    int32_t exponent_bias = 0;

    // Konwersja wejścia na format 1.16
    if (temp_exponent < 0) {
        mant_val = 0;
        mant_val.set_slc(9, (ac_int<7, false>)input_parts.mantissa);
        mant_val[16] = 1;
        mant_val >>= (-temp_exponent);
        exponent_bias = 0;
    } else {
        typedef ac_fixed<15, 8, false> fix_8_7_t;
        fix_8_7_t val = 0;
        val.set_slc(0, (ac_int<7, false>)input_parts.mantissa);
        val[7] = 1;
        val <<= temp_exponent;
        mant_val = 0;
        mant_val.set_slc(9, val.slc<7>(0));
        exponent_bias = input_parts.sign ? -(int)val.slc<8>(7).to_int() : (int)val.slc<8>(7).to_int();
    }

    PolyResult poly_res = bf16_exp2_poly(mant_val, input_parts.sign);

    int32_t final_exponent = poly_res.exponent + exponent_bias;
    ac_int<41, false> full_mant = poly_res.mantissa.slc<41>(0);
    
    int shift = 0;
    bool is_denormal_range = (final_exponent < -126);

    if (is_denormal_range) {
        // MSB (waga 2^final_exponent) musi trafić na pozycję odpowiadającą 
        // jego wadze względem najmniejszego denormala 2^-133.
        shift = 40 - (final_exponent + 133);
    } else {
        // Liczba normalna: format 1.40 -> 1.7 (przesunięcie o 33 bity)
        shift = 33;
    }

    uint32_t result_m = 0;
    bool guard = false;
    bool sticky = false;
    bool lsb = false;

    if (shift >= 42) {
        sticky = (full_mant != 0);
    } else if (shift > 0) {
        result_m = (full_mant >> shift).to_int();
        ac_int<41, false> mask = (ac_int<41, false>(1) << shift) - 1;
        ac_int<41, false> remainder = full_mant & mask;
        guard = (remainder >> (shift - 1)) & 1;
        sticky = (remainder & ((ac_int<41, false>(1) << (shift - 1)) - 1)) != 0;
        lsb = result_m & 1;
    } else {
        result_m = full_mant.to_int();
    }

    if (guard && (sticky || lsb)) {
        result_m++;
    }

    result.sign = 0;
    if (is_denormal_range) {
        if (result_m > 127) {
            result.mantissa = 0;
            result.hidden_bit = 1;
            result.exponent = -126;
            result.status.is_denormal = false;
        } else if (result_m == 0) {
            result.status.is_zero = true;
            result.exponent = 0;
        } else {
            result.mantissa = result_m;
            result.hidden_bit = 0;
            result.exponent = -127; 
            result.status.is_denormal = true;
        }
    } else {
        if (result_m > 255) {
            result.mantissa = 0;
            result.hidden_bit = 1;
            result.exponent = final_exponent + 1;
        } else {
            result.mantissa = result_m & 0x7F;
            result.hidden_bit = 1;
            result.exponent = final_exponent;
        }
    }
    
    return result;
}

#endif // BF16_EXP2_CORE_HPP
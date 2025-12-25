#ifndef BF16_EXP2_CORE_HPP
#define BF16_EXP2_CORE_HPP

#include "../utils/fp_utils.hpp"
#include "../../modeling/coeff_gen/bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"

// Parametryzacja formatów stałoprzecinkowych
namespace bf16_cfg {
    // Wejście (format 1.16 unsigned)
    constexpr int IN_I = 1;
    constexpr int IN_F = 16;
    constexpr int IN_W = IN_I + IN_F;

    // Parametry LUT
    constexpr int LUT_ADDR_W = 6;               // Liczba bitów adresu (log2 z rozmiaru LUT)
    constexpr int LUT_SIZE = 1 << LUT_ADDR_W;   // Rozmiar tablicy (np. 64)
    constexpr int LUT_MAX_IDX = LUT_SIZE - 1;   // Maksymalny indeks (np. 63)

    // Współczynniki (format 1.25 unsigned)
    constexpr int COEFF_I = 1;
    constexpr int COEFF_F = 25;
    constexpr int COEFF_W = COEFF_I + COEFF_F;

    // 1. Parametry mnożenia (a * x) -> wynik unsigned
    // Szerokość bitowa iloczynu to suma szerokości czynników
    constexpr int MULT_I = IN_I + COEFF_I; // 1 + 1 = 2
    constexpr int MULT_F = IN_F + COEFF_F; // 16 + 25 = 41
    constexpr int MULT_W = MULT_I + MULT_F;

    constexpr int CALC_SIGN_BIT = 1;  // Bit znaku (wymagany dla liczb ze znakiem)


    // 2. Obliczenia pośrednie (suma b - ax) -> wynik signed
    // Wybieramy większą szerokość całkowitą i ułamkową spośród składników (ax i b)
    // Dzięki temu format CALC zawsze pomieści "szerszy" z operandów
    constexpr int MAX_OP_I = (MULT_I + CALC_SIGN_BIT > COEFF_I) ? MULT_I + CALC_SIGN_BIT : COEFF_I;
    constexpr int MAX_OP_F = (MULT_F > COEFF_F) ? MULT_F : COEFF_F;

    constexpr int CALC_ADD_GUARD = 1; // Bit na przeniesienie (overflow) przy dodawaniu
    
    // CALC_I: Największa część całkowita + miejsce na znak i przeniesienie
    constexpr int CALC_I = MAX_OP_I + CALC_ADD_GUARD; 
    
    // CALC_F: Największa precyzja ułamkowa (aby nie tracić bitów przed wynikiem)
    constexpr int CALC_F = MAX_OP_F;
    
    constexpr int CALC_W = CALC_I + CALC_F;

    // Wynik wielomianu (format 1.41 unsigned - 41 bitów ułamkowych dla RNE)
    constexpr int POLY_OUT_I = 1;
    constexpr int POLY_OUT_F = CALC_F;
    constexpr int POLY_OUT_W = POLY_OUT_I + POLY_OUT_F;
}

// Typ stałoprzecinkowy wejściowy
typedef ac_fixed<bf16_cfg::IN_W, bf16_cfg::IN_I, false> mant_t;

struct PolyResult {
    ac_fixed<bf16_cfg::POLY_OUT_W, bf16_cfg::POLY_OUT_I, false> mantissa; 
    int32_t exponent;
};

/**
 * @brief Oblicza 2^(-x) przy użyciu aproksymacji odcinkowo-liniowej dla x in [0, 1].
 * Formuła: res = a * (-x) + b
 */
inline PolyResult bf16_exp2_poly(mant_t mant_val) {
    // Indeks LUT pobierany z najstarszych bitów części ułamkowej
    uint8_t lut_index = mant_val.slc<bf16_cfg::LUT_ADDR_W>(bf16_cfg::IN_F - bf16_cfg::LUT_ADDR_W);

    float a_float = bf16_exp2::coeffs_a[bf16_cfg::LUT_MAX_IDX - lut_index];
    float b_float = bf16_exp2::coeffs_b[bf16_cfg::LUT_MAX_IDX - lut_index];

    // Definicja typów obliczeniowych na podstawie parametrów
    typedef ac_fixed<bf16_cfg::COEFF_W, bf16_cfg::COEFF_I, false> coeff_t;
    typedef ac_fixed<bf16_cfg::CALC_W, bf16_cfg::CALC_I, true> calc_t;

    coeff_t a_fixed = a_float;
    coeff_t b_fixed = b_float;

    // Mnożenie: a * x (format wynikowy zdefiniowany w cfg jako MULT)
    ac_fixed<bf16_cfg::MULT_W, bf16_cfg::MULT_I, false> ax_u = a_fixed * mant_val;

    // Negacja i rzutowanie na typ signed: -ax
    calc_t ax_s = - (calc_t)ax_u;
    
    // Dodanie b: res = b + (-ax)
    calc_t res = ax_s + (calc_t)b_fixed;
    
    // Szukanie pozycji MSB dla normalizacji
    int msb_idx = -1;
    for (int i = bf16_cfg::CALC_W - 1; i >= 0; --i) {
        if (res[i]) {
            msb_idx = i;
            break;
        }
    }

    PolyResult result;
    // Wykładnik względem formatu 1.41 (jeśli MSB na pozycji bitu CALC_F, wykładnik = 0)
    result.exponent = msb_idx - bf16_cfg::POLY_OUT_F;

    // Wycięcie 42 bitów (MSB + 41 ułamkowych) dla zapewnienia precyzji RNE
    ac_int<bf16_cfg::POLY_OUT_W, false> raw_bits = 0;
    for (int i = 0; i < bf16_cfg::POLY_OUT_W; ++i) {
        int bit_idx = msb_idx - i;
        if (bit_idx >= 0 && res[bit_idx]) {
            raw_bits[bf16_cfg::POLY_OUT_W - 1 - i] = 1;
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
        mant_val.set_slc(bf16_cfg::IN_F - 7, (ac_int<7, false>)input_parts.mantissa);
        mant_val[bf16_cfg::IN_F] = 1; // Hidden bit
        mant_val >>= (-temp_exponent);
        exponent_bias = 0;
    } else {
        typedef ac_fixed<15, 8, false> fix_8_7_t;
        fix_8_7_t val = 0;
        val.set_slc(0, (ac_int<7, false>)input_parts.mantissa);
        val[7] = 1;
        val <<= temp_exponent;
        mant_val = 0;
        mant_val.set_slc(bf16_cfg::IN_F - 7, val.slc<7>(0));
        // Wykładnik całkowity liczby wejściowej, zawsze będzie ujemny bo x są ujemne ( lub -0)
        exponent_bias = -(int)val.slc<8>(7).to_int();
    }

    // Wywołanie wielomianu (bez znaku - logika przesunięta do środka poly)
    PolyResult poly_res = bf16_exp2_poly(mant_val);

    int32_t final_exponent = poly_res.exponent + exponent_bias;
    ac_int<bf16_cfg::POLY_OUT_W, false> full_mant = poly_res.mantissa.slc<bf16_cfg::POLY_OUT_W>(0);
    
    int shift = 0;
    bool is_denormal_range = (final_exponent < -126);

    // Dynamiczne obliczanie shift na podstawie parametrów szerokości
    if (is_denormal_range) {
        shift = (bf16_cfg::POLY_OUT_W - 1) - (final_exponent + 133);
    } else {
        // Normalna liczba: przesunięcie z formatu 1.40 do formatu 1.7 (BF16)
        shift = (bf16_cfg::POLY_OUT_W - 1) - 7; 
    }

    uint32_t result_m = 0;
    bool guard = false;
    bool sticky = false;
    bool lsb = false;

    if (shift >= (bf16_cfg::POLY_OUT_W + 1)) {
        sticky = (full_mant != 0);
    } else if (shift > 0) {
        result_m = (full_mant >> shift).to_int();
        ac_int<bf16_cfg::POLY_OUT_W, false> mask = (ac_int<bf16_cfg::POLY_OUT_W, false>(1) << shift) - 1;
        ac_int<bf16_cfg::POLY_OUT_W, false> remainder = full_mant & mask;
        guard = (remainder >> (shift - 1)) & 1;
        sticky = (remainder & ((ac_int<bf16_cfg::POLY_OUT_W, false>(1) << (shift - 1)) - 1)) != 0;
        lsb = result_m & 1;
    } else {
        result_m = full_mant.to_int();
    }

    // Zaokrąglanie RNE
    if (guard && (sticky || lsb)) {
        result_m++;
    }

    result.sign = 0;
    if (is_denormal_range) {
        if (result_m > 127) { // Overflow z denormal do normal
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
        if (result_m > 255) { // Overflow mantysy (carry out)
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
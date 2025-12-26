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


    // --- Parametry formatu docelowego (BF16) ---
    constexpr int TARGET_MANT_W = 7;
    constexpr int TARGET_EXP_BIAS = 127;
    constexpr int TARGET_MIN_EXP = 1 - TARGET_EXP_BIAS; // -126 dla BF16

    // --- Logika wyrównywania i zaokrąglania ---
    // Bazowe przesunięcie: różnica między precyzją wielomianu a mantysą docelową
    // 1.41 -> 1.7 wymaga przesunięcia o 34 bity (41 - 7), aby LSB się pokryły
    constexpr int BASE_SHIFT = POLY_OUT_F - TARGET_MANT_W; 
    
    // Szerokość rozszerzonej mantysy: bit przeniesienia + hidden bit + mantysa
    constexpr int EXT_MANT_W = TARGET_MANT_W + 2;    // 7 + 1 + 1 = 9
    constexpr int CARRY_BIT_IDX = EXT_MANT_W - 1;    // Indeks bitu overflow (8)
    constexpr int HIDDEN_BIT_IDX = TARGET_MANT_W;    // Indeks bitu ukrytego (7)

    // --- Parametry konwersji wejściowej ---
    constexpr int IN_CONV_INT_W = 8;                 // Bity na część całkowitą x
    constexpr int IN_CONV_FRAC_W = TARGET_MANT_W;    // Bity na część ułamkową x
    constexpr int IN_CONV_W = IN_CONV_INT_W + IN_CONV_FRAC_W;
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
    
    // Rzutowanie na ac_int pozwala na traktowanie liczby jako surowego ciągu bitów
    ac_int<bf16_cfg::CALC_W, false> res_raw = res.slc<bf16_cfg::CALC_W>(0);

    // 1. Priority Encoder (szukanie MSB)
    // To jest standardowy wzorzec HLS, który syntezuje się do szybkiej logiki kombinacyjnej
    int msb_idx = -1;
    for (int i = bf16_cfg::CALC_W - 1; i >= 0; --i) {
        if (res_raw[i]) {
            msb_idx = i;
            break;
        }
    }

    PolyResult result;
    // Wykładnik względem formatu 1.41
    result.exponent = msb_idx - bf16_cfg::POLY_OUT_F;

    // 2. Normalizacja (Barrel Shifter + Slice)
    // Zamiast pętli kopiującej, używamy przesunięcia bitowego, aby wyrównać MSB do lewej strony
    // Obliczamy o ile trzeba przesunąć w lewo, aby MSB trafił na pozycję (CALC_W - 1)
    int shift = (bf16_cfg::CALC_W - 1) - msb_idx;
    
    // Przesunięcie (Barrel Shifter)
    ac_int<bf16_cfg::CALC_W, false> normalized = res_raw << shift;

    // Wycięcie (Slice) - pobieramy POLY_OUT_W najstarszych bitów
    // Indeks początkowy to: (Szerokość całkowita) - (Szerokość docelowa)
    result.mantissa.set_slc(0, normalized.slc<bf16_cfg::POLY_OUT_W>(bf16_cfg::CALC_W - bf16_cfg::POLY_OUT_W));
    
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
        // Wstawienie mantysy na odpowiednie pozycje względem IN_F
        mant_val.set_slc(bf16_cfg::IN_F - bf16_cfg::TARGET_MANT_W, (ac_int<bf16_cfg::TARGET_MANT_W, false>)input_parts.mantissa);
        mant_val[bf16_cfg::IN_F] = 1; // Hidden bit
        mant_val >>= (-temp_exponent);
        exponent_bias = 0;
    } else {
        // Pomocniczy typ stałoprzecinkowy do rozbicia wejścia na część całkowitą i ułamkową
        typedef ac_fixed<bf16_cfg::IN_CONV_W, bf16_cfg::IN_CONV_INT_W, false> in_fix_t;
        in_fix_t val = 0;
        val.set_slc(0, (ac_int<bf16_cfg::TARGET_MANT_W, false>)input_parts.mantissa);
        val[bf16_cfg::TARGET_MANT_W] = 1;
        val <<= temp_exponent;
        
        mant_val = 0;
        mant_val.set_slc(bf16_cfg::IN_F - bf16_cfg::IN_CONV_FRAC_W, val.slc<bf16_cfg::IN_CONV_FRAC_W>(0));
        
        // Wykładnik całkowity liczby wejściowej determinuje przesunięcie końcowe
        exponent_bias = -(int)val.slc<bf16_cfg::IN_CONV_INT_W>(bf16_cfg::IN_CONV_FRAC_W).to_int();
    }

    // Wywołanie aproksymacji wielomianowej
    PolyResult poly_res = bf16_exp2_poly(mant_val);

    int32_t final_exponent = poly_res.exponent + exponent_bias;
    ac_int<bf16_cfg::POLY_OUT_W, false> full_mant = poly_res.mantissa.slc<bf16_cfg::POLY_OUT_W>(0);
    
    // 1. Wyznaczenie parametrów wyrównania (Alignment Logic)
    bool is_sub = (final_exponent < bf16_cfg::TARGET_MIN_EXP);
    
    // shift_val to liczba bitów "odrzucanych" w prawo przy zaokrąglaniu
    int shift_val = bf16_cfg::BASE_SHIFT + (is_sub ? (bf16_cfg::TARGET_MIN_EXP - final_exponent) : 0);

    // 2. Ekstrakcja bitów do zaokrąglenia (Zabezpieczona logika RNE)
    ac_int<bf16_cfg::POLY_OUT_W, false> m_raw = full_mant;
    
    bool lsb_bit = (shift_val < bf16_cfg::POLY_OUT_W) ? (bool)m_raw[shift_val] : false;
    bool guard_bit = (shift_val > 0 && shift_val <= bf16_cfg::POLY_OUT_W) ? (bool)m_raw[shift_val - 1] : false;
    
    bool sticky_bit = false;
    if (shift_val > 1) {
        if (shift_val > bf16_cfg::POLY_OUT_W) {
            sticky_bit = (m_raw != 0); 
        } else {
            ac_int<bf16_cfg::POLY_OUT_W, false> mask = (ac_int<bf16_cfg::POLY_OUT_W, false>(1) << (shift_val - 1)) - 1;
            sticky_bit = (m_raw & mask) != 0;
        }
    }

    bool round_up = guard_bit && (lsb_bit || sticky_bit);

    // 3. Przesunięcie i zaokrąglenie
    ac_int<bf16_cfg::EXT_MANT_W, false> result_m_ext = 0;
    if (shift_val < bf16_cfg::POLY_OUT_W) {
        result_m_ext = (ac_int<bf16_cfg::EXT_MANT_W, false>)(m_raw >> shift_val);
    }
    if (round_up) result_m_ext++;

    // 4. Normalizacja i korekta wykładnika (obsługa carry po zaokrągleniu)
    int32_t adjusted_exp = is_sub ? bf16_cfg::TARGET_MIN_EXP : final_exponent;
    if (result_m_ext[bf16_cfg::CARRY_BIT_IDX]) {
        adjusted_exp++;
        result_m_ext >>= 1;
    }

    // 5. Finalne formowanie struktury
    result.sign = 0;
    if (result_m_ext == 0) {
        result.status.is_zero = true;
        result.exponent = 0;
    } else if (is_sub && !result_m_ext[bf16_cfg::HIDDEN_BIT_IDX]) {
        result.mantissa = result_m_ext.slc<bf16_cfg::TARGET_MANT_W>(0);
        result.hidden_bit = 0;
        result.exponent = bf16_cfg::TARGET_MIN_EXP - 1; // -127 dla BF16
        result.status.is_denormal = true;
    } else {
        result.mantissa = result_m_ext.slc<bf16_cfg::TARGET_MANT_W>(0);
        result.hidden_bit = 1;
        result.exponent = adjusted_exp;
        result.status.is_denormal = false;
    }
    
    return result;
}

#endif // BF16_EXP2_CORE_HPP
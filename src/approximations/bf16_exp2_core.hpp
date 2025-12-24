#ifndef BF16_EXP2_CORE_HPP
#define BF16_EXP2_CORE_HPP

#include "../utils/fp_utils.hpp"
#include "../../modeling/coeff_gen/bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"

/**
 * @brief Core approximation logic for exp2(x) in exponent range [-9, 7] (only negative inputs).
 */
inline FPRaw bf16_exp2_core_approx(const FPRaw& input_parts) {
    FPRaw result = {};
    
    // Typ stałoprzecinkowy: 1 bit całkowity (dla wartości < 1.0), 16 ułamkowych.
    // Zakładamy, że pracujemy na wartości bezwzględnej |x| przed przejściem na typ ze znakiem.
    typedef ac_fixed<17, 1, false> mant_t;
    mant_t mant_val;

    int32_t temp_exponent = input_parts.exponent;

    // 1. Obsługa przypadku zero (2^0 = 1)
    if (input_parts.status.is_zero) {
        result.exponent = 0;
        result.mantissa = 0;
        result.hidden_bit = 1;
        result.sign = 0;
        return result;
    }

    // 2. Konwersja wejścia na fixed-point w zależności od zakresu wykładnika
    if (temp_exponent < 0) {
        // Dla temp_exponent < 0, |x| mieści się w przedziale (0, 1).
        mant_val = 0;
        
        // Wyciągamy bity ułamkowe BF16 i umieszczamy je w mant_t.
        // Mantysa BF16 (7 bitów) po doklejeniu hidden bitu tworzy format 1.xxxxxxx.
        // W mant_t <17, 1> bit o indeksie 16 to bit całkowity, bity 15-0 to ułamkowe.
        
        // Wstawiamy bity mantysy (7 bitów) na najwyższe pozycje ułamkowe (indeksy 15 do 9)
        mant_val.set_slc(9, (ac_int<7, false>)input_parts.mantissa);
        
        // Doklejamy hidden bit (zawsze 1 dla liczb znormalizowanych/obsługiwanych tutaj)
        mant_val[16] = 1;

        // Przesunięcie w prawo o abs(temp_exponent) realizuje mnożenie przez 2^exp.
        // Przykład: dla exp = -1, mant_val 1.0 zmienia się w 0.1 (binarnie), czyli 0.5 dziesiętnie.
        mant_val >>= (-temp_exponent);
    }
    else {
        // Dla wartości |x| >= 1 (wykładniki >= 0), na razie ustawiamy stałą wartość 1.0
        mant_val = 1.0;
    }

    // 3. Pobranie indeksu do LUT (6 najstarszych bitów ułamkowych: 15, 14, 13, 12, 11, 10)
    uint8_t lut_index = mant_val.slc<6>(10);

    // 4. Pobranie współczynników a i b
    float a_float = bf16_exp2::coeffs_a[63 - lut_index];
    float b_float = bf16_exp2::coeffs_b[63 - lut_index];

    typedef ac_fixed<32, 8, true> coeffs_t;
    typedef ac_fixed<56, 16, true> calc_t;
    coeffs_t a_fixed = a_float;
    coeffs_t b_fixed = b_float;

    // 5. Obliczenie ax + b
    // Ponieważ funkcja obsługuje x < 0, nadajemy x_fixed znak ujemny
    coeffs_t x_fixed = mant_val;
    if (input_parts.sign) {
        x_fixed = -x_fixed;
    }

    calc_t ax = a_fixed * x_fixed;
    calc_t res = ax + b_fixed;
    
    // 6. Konwersja wyniku (fixed-point) na FPRaw
    int msb_idx = -1;
    for (int i = 54; i >= 0; --i) {
        if (res[i]) {
            msb_idx = i;
            break;
        }
    }

    if (msb_idx == -1) {
        result.status.is_zero = true;
        return result;
    }

    // Obliczenie eksponenty: waga bitu k to 2^(k - 40) (16 bitów całkowitych w calc_t)
    result.exponent = msb_idx - 40;

    // Pobranie 7 bitów mantysy
    uint32_t mantissa = 0;
    for (int i = 1; i <= 7; ++i) {
        int bit_idx = msb_idx - i;
        if (bit_idx >= 0 && res[bit_idx]) {
            mantissa |= (1 << (7 - i));
        }
    }

    // Zaokrąglenie RNE
    int lsb_idx = msb_idx - 7;
    int guard_idx = msb_idx - 8;
    bool lsb_bit = (lsb_idx >= 0) ? (bool)res[lsb_idx] : false;
    bool guard_bit = (guard_idx >= 0) ? (bool)res[guard_idx] : false;
    bool sticky_bit = false;
    for (int k = 0; k < guard_idx; ++k) {
        if (res[k]) {
            sticky_bit = true;
            break;
        }
    }

    if (guard_bit && (sticky_bit || lsb_bit)) {
        mantissa++;
        if (mantissa > 127) {
            mantissa = 0;
            result.exponent++;
        }
    }

    result.sign = 0;
    result.mantissa = mantissa;
    result.hidden_bit = 1;
    
    return result;
}
#endif // BF16_EXP2_CORE_HPP
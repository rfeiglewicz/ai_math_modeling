#ifndef BF16_EXP2_CORE_HPP
#define BF16_EXP2_CORE_HPP

#include "../utils/fp_utils.hpp"
#include "../../modeling/coeff_gen/bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"

/**
 * @brief Core approximation logic for exp2(x) in range [-9, 7].
 * 
 * This function uses ac_datatypes for fixed-point arithmetic.
 * 
 * @param input_parts Decomposed BF16 input
 * @return FPRaw Decomposed BF16 result
 */
inline FPRaw bf16_exp2_core_approx(const FPRaw& input_parts) {
    FPRaw result = {};
    
    // Definiujemy typ stałoprzecinkowy: 24 bity szerokości, 8 bitów całkowitych, bez znaku.
    // 8 bitów całkowitych pozwala na reprezentację liczby >= 1 oraz bezpieczne przesunięcia.
    // 16 bitów ułamkowych zapewnia zapas precyzji względem 7 bitów mantysy BF16.
    typedef ac_fixed<24, 8, false> mant_t;

    int32_t temp_exponent = input_parts.exponent;
    
    // 1. Inicjalizacja mantysy wartością całkowitą z bitów (0..127)
    mant_t mant_val = input_parts.mantissa;
    
    // 2. Przesunięcie w prawo o 7, aby zinterpretować bity jako ułamek .xxxxxxx
    // (odpowiednik dzielenia przez 128)
    mant_val >>= 7;

    // 3. Dodanie ukrytego bitu (1.xxxxxxx)
    if (input_parts.hidden_bit) {
        mant_val += 1;
    }

    // 4. Obsługa ujemnej eksponenty (denormalizacja do zakresu fixed-point)
    // Przesuwamy mantysę w prawo i inkrementujemy eksponentę, aż osiągnie 0.
    while (temp_exponent < 0) {
        mant_val >>= 1;
        temp_exponent++;
    }
    
    // 5. Pobranie indeksu do LUT (6 najstarszych bitów ułamkowych)
    // mant_val (ac_fixed<24,8>) ma 16 bitów ułamkowych (indeksy 0..15).
    // Najstarsze 6 bitów to indeksy 15, 14, 13, 12, 11, 10.
    // Używamy metody slc (slice) z ac_datatypes.
    uint8_t lut_index = mant_val.slc<6>(10);

    // 6. Pobranie współczynników a i b z LUT i konwersja na fixed-point
    float a_float = bf16_exp2::coeffs_a[63 - lut_index];
    float b_float = bf16_exp2::coeffs_b[63 - lut_index];

    // Typ do obliczeń ze znakiem (ax + b)
    typedef ac_fixed<32, 8, true> coeffs_t;
    typedef ac_fixed<56, 16, true> calc_t;
    coeffs_t a_fixed = a_float;
    coeffs_t b_fixed = b_float;

    // 7. Obliczenie ax + b
    // Konwersja mantysy na typ ze znakiem (x jest ujemne jeśli input_parts.sign == 1)
    coeffs_t x_fixed = mant_val;
    if (input_parts.sign) {
        x_fixed = -x_fixed;
    }

    calc_t ax = a_fixed * x_fixed;
    calc_t res = ax + b_fixed;
    
    // 8. Konwersja wyniku (fixed-point) na FPRaw
    // Szukamy pierwszego bitu 1 (MSB) aby znormalizować wynik
    int msb_idx = -1;
    // calc_t ma 56 bitów (indeksy 0..55). Bit 55 to znak.
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

    // Obliczenie eksponenty
    // Waga bitu k to 2^(k - 40) (bo 16 bitów całkowitych, 56-16=40 ułamkowych)
    result.exponent = msb_idx - 40;

    // Pobranie 7 bitów mantysy (bity po MSB)
    uint32_t mantissa = 0;
    for (int i = 1; i <= 7; ++i) {
        int bit_idx = msb_idx - i;
        if (bit_idx >= 0 && res[bit_idx]) {
            mantissa |= (1 << (7 - i));
        }
    }

    // Round to Nearest Even (RNE)
    int lsb_idx = msb_idx - 7;
    int guard_idx = msb_idx - 8;
    
    bool lsb_bit = (lsb_idx >= 0) ? static_cast<bool>(res[lsb_idx]) : false;
    bool guard_bit = (guard_idx >= 0) ? static_cast<bool>(res[guard_idx]) : false;
    bool sticky_bit = false;
    
    // Sprawdzenie bitów "sticky" (wszystkie bity poniżej guard bit)
    for (int k = 0; k < guard_idx; ++k) {
        if (res[k]) {
            sticky_bit = true;
            break;
        }
    }

    // Warunek zaokrąglenia w górę:
    // 1. Guard bit jest 1 ORAZ (Sticky bit jest 1 LUB LSB jest 1)
    if (guard_bit && (sticky_bit || lsb_bit)) {
        mantissa++;
        // Obsługa przepełnienia mantysy (np. 1.11...1 -> 10.00...0)
        if (mantissa > 127) {
            mantissa = 0;
            result.exponent++;
        }
    }

    result.sign = 0;        // Result of exp2 is always positive
    result.mantissa = mantissa;
    result.hidden_bit = 1;
    
    return result;
}

#endif // BF16_EXP2_CORE_HPP
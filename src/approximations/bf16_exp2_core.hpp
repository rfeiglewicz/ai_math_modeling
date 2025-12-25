#ifndef BF16_EXP2_CORE_HPP
#define BF16_EXP2_CORE_HPP

#include "../utils/fp_utils.hpp"
#include "../../modeling/coeff_gen/bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"

// Typ stałoprzecinkowy: 1 bit całkowity (dla wartości < 1.0), 16 ułamkowych.
typedef ac_fixed<17, 1, false> mant_t;

struct PolyResult {
    ac_fixed<8, 1, false> mantissa; // Format 1.7 (1 bit int, 7 bits frac)
    int32_t exponent;
};

/**
 * @brief Helper function: Calculates 2^x using piecewise linear approximation.
 * 
 * @param mant_val Input value in fixed-point format (absolute value)
 * @param sign Input sign (true if negative)
 * @return PolyResult Normalized result in 1.7 fixed point and unbiased exponent
 */
inline PolyResult bf16_exp2_poly(mant_t mant_val, bool sign) {
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
    if (sign) {
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

    PolyResult result;
    // Obliczenie eksponenty: waga bitu k to 2^(k - 40) (16 bitów całkowitych w calc_t)
    result.exponent = msb_idx - 40;

    // Pobranie 7 bitów mantysy
    uint32_t mantissa_frac = 0;
    for (int i = 1; i <= 7; ++i) {
        int bit_idx = msb_idx - i;
        if (bit_idx >= 0 && res[bit_idx]) {
            mantissa_frac |= (1 << (7 - i));
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
        mantissa_frac++;
        if (mantissa_frac > 127) {
            mantissa_frac = 0;
            result.exponent++;
        }
    }

    // Złożenie wyniku w formacie 1.7 (ac_fixed<8, 1>)
    // Bit 7 (MSB) to bit całkowity (hidden bit = 1), bity 6-0 to ułamek
    ac_int<8, false> raw_bits = (1 << 7) | mantissa_frac;
    result.mantissa.set_slc(0, raw_bits);
    
    return result;
}

/**
 * @brief Core approximation logic for exp2(x) in exponent range [-9, 7] (only negative inputs).
 */
inline FPRaw bf16_exp2_core_approx(const FPRaw& input_parts) {
    FPRaw result = {};
    
    mant_t mant_val;
    int32_t temp_exponent = input_parts.exponent;
    int32_t exponent_bias = 0;

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
        
        exponent_bias = 0;
    }
    else {
        // Deklaracja typu stałoprzecinkowego 8.7 (8 bitów całkowitych, 7 ułamkowych)
        typedef ac_fixed<15, 8, false> fix_8_7_t;
        fix_8_7_t val = 0;

        // Wprowadzenie mantysy (7 bitów) na pozycje ułamkowe
        val.set_slc(0, (ac_int<7, false>)input_parts.mantissa);
        // Ustawienie hidden bitu (bit o wadze 2^0, indeks 7)
        val[7] = 1;

        // Przesunięcie w lewo o wartość eksponenty (uzyskanie rzeczywistej wartości |x|)
        val <<= temp_exponent;

        // Wyodrębnienie części całkowitej (będzie dodana do eksponenty wyniku)
        // Użycie slc zamiast to_int
        ac_int<8, false> int_part_bits = val.slc<8>(7);
        int integer_part = int_part_bits.to_int();

        // Wyodrębnienie części ułamkowej i przypisanie do mant_val
        // mant_val (format 1.16) oczekuje ułamka na bitach 15-0.
        // Bity ułamkowe z val (indeksy 6-0) trafiają na starsze bity ułamkowe mant_val (15-9).
        mant_val = 0;
        mant_val.set_slc(9, val.slc<7>(0));

        // Zapisanie części całkowitej w exponent_bias (z uwzględnieniem znaku)
        exponent_bias = input_parts.sign ? -integer_part : integer_part;
    }

    // Wywołanie funkcji obliczającej wielomian/aproksymację
    PolyResult poly_res = bf16_exp2_poly(mant_val, input_parts.sign);

    result.sign = 0;
    result.exponent = poly_res.exponent + exponent_bias;
    
    // Konwersja z formatu 1.7 (ac_fixed<8,1>) na FPRaw
    // FPRaw oczekuje samej części ułamkowej w polu mantissa (7 bitów)
    result.mantissa = poly_res.mantissa.slc<7>(0);
    // Bit całkowity (hidden bit)
    result.hidden_bit = poly_res.mantissa[7];
    
    // Obsługa denormów
    // BF16: bias 127. Min normal exp = -126.
    if (result.exponent < -126) {
        int shift = -126 - result.exponent;
        uint16_t full_mant = (result.hidden_bit << 7) | result.mantissa;
        uint16_t result_m = 0;

        if (shift > 8) {
            result_m = 0;
        } else {
            result_m = full_mant >> shift;
            
            // Rounding (RNE)
            uint16_t remainder = full_mant & ((1 << shift) - 1);
            if (shift > 0) {
                bool guard = (remainder >> (shift - 1)) & 1;
                bool sticky = (remainder & ((1 << (shift - 1)) - 1)) != 0;
                bool lsb = result_m & 1;
                
                if (guard && (sticky || lsb)) {
                    result_m++;
                }
            }
        }
        
        if (result_m > 127) {
            // Rounding caused overflow back to normal
            result.mantissa = 0;
            result.hidden_bit = 1;
            result.exponent = -126;
        } else {
            result.mantissa = result_m;
            result.hidden_bit = 0;
            result.exponent = -127; // Encodes to 0 in biased representation
        }
    }
    
    return result;
}
#endif // BF16_EXP2_CORE_HPP
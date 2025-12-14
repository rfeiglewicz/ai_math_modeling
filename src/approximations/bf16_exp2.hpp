#ifndef BF16_EXP2_HPP
#define BF16_EXP2_HPP

#include "../utils/fp_utils.hpp"
#include <cstdint>

/**
 * @brief Custom approximation of exp2(x) for BF16.
 * 
 * Logic:
 * 1. Special Cases (Priority):
 *    - NaN -> NaN
 *    - +Inf -> +Inf
 *    - -Inf -> 0
 *    - +/-0 -> 1
 * 2. Negative Inputs (x < 0):
 *    - Exp < -9: Return 0.25 (1/4)
 *    - Else:     Return 0.5 (1/2)
 * 3. Positive Inputs (x > 0):
 *    - Exp < -8: Return 1.0
 *    - Exp > 6:  Return +Inf
 *    - Else:     Return 2.0
 * 
 * @param raw_input Raw 16-bit BF16 payload
 * @return Raw 16-bit BF16 result
 */
inline uint16_t bf16_exp2_approx(uint16_t raw_input) {
    // 1. Decompose input
    FPRaw input_parts = fp_decompose(static_cast<uint32_t>(raw_input), FPType::BF16);

    // Prepare result structure
    FPRaw result_parts = {};
    result_parts.sign = false; // 2^x is always positive (unless NaN)

    // --- LOGIC START ---

    if (input_parts.status.is_nan) {
        // NaN -> NaN (propagate payload if desired, here just standard QNaN)
        result_parts.status.is_nan = true;
        result_parts.mantissa = input_parts.mantissa;
        // Sign of NaN usually doesn't matter, but we can preserve it
        result_parts.sign = input_parts.sign; 
    }
    else if (input_parts.status.is_zero) {
        // 2^0 = 1 (regardless of sign of zero)
        result_parts.exponent = 0;
        result_parts.mantissa = 0;
        result_parts.hidden_bit = 1;
    }
    else if (input_parts.status.is_inf) {
        if (input_parts.sign) {
            // 2^(-inf) = 0
            result_parts.status.is_zero = true;
        } else {
            // 2^(+inf) = +inf
            result_parts.status.is_inf = true;
        }
    }
    else {
        // --- HANDLE NORMAL/DENORMAL NUMBERS ---
        if (input_parts.sign) {
            // Case: Negative inputs (x < 0)
            int32_t x_exp = input_parts.exponent;

            if (x_exp < -9) {
                // Very small negative numbers -> Return 0.25 (1/4)
                // 0.25 is Exp=-2 (biased 125), Mant=0
                result_parts.exponent = -2;
                result_parts.mantissa = 0;
                result_parts.hidden_bit = 1;
            } else {
                // Larger negative numbers -> Return 0.5 (1/2)
                // 0.5 is Exp=-1 (biased 126), Mant=0
                result_parts.exponent = -1; 
                result_parts.mantissa = 0;
                result_parts.hidden_bit = 1;
            }
        } 
        else {
            // Case: Positive inputs (x > 0)
            int32_t x_exp = input_parts.exponent;

            if (x_exp < -8) {
                // Small positive numbers -> Return 1.0
                result_parts.exponent = 0;
                result_parts.mantissa = 0;
                result_parts.hidden_bit = 1;
            } 
            else if (x_exp > 6) {
                // Large positive numbers -> Return +Inf
                result_parts.status.is_inf = true;
            } 
            else {
                // Range [-8, 6] -> Return 2.0
                result_parts.exponent = 1;
                result_parts.mantissa = 0;
                result_parts.hidden_bit = 1;
            }
        }
    }

    // --- LOGIC END ---

    // 3. Recompose result
    uint32_t result_32 = fp_recompose(result_parts, FPType::BF16);
    return static_cast<uint16_t>(result_32);
}

#endif // BF16_EXP2_HPP
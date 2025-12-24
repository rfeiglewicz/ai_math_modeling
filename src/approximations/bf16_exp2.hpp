#ifndef BF16_EXP2_HPP
#define BF16_EXP2_HPP

#include "../utils/fp_utils.hpp"
#include "bf16_exp2_core.hpp"
#include <cstdint>

/**
 * @brief Custom approximation of exp2(x) for BF16.
 * 
 * Logic:
 * 1. Special Cases:
 *    - NaN -> NaN
 *    - -Inf -> 0
 *    - +/-0 -> 1
 * 2. Positive Inputs (x > 0):
 *    - Always Return 1.0
 * 3. Negative Inputs (x < 0):
 *    - Exp < -9: Return 1.0
 *    - Exp > 7:  Return +0.0
 *    - Exp [-9, 7]: Placeholder for approximation
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
        // NaN -> NaN
        result_parts.status.is_nan = true;
        result_parts.mantissa = input_parts.mantissa;
        result_parts.sign = input_parts.sign; 
    }
    else if (input_parts.status.is_zero) {
        // 2^0 = 1
        result_parts.exponent = 0;
        result_parts.mantissa = 0;
        result_parts.hidden_bit = 1;
    }
    else if (input_parts.status.is_inf) {
        if (input_parts.sign) {
            // 2^(-inf) = 0
            result_parts.status.is_zero = true;
        } else {
            // 2^(+inf) -> 1.0 (per requirement "positive values always return 1")
            result_parts.exponent = 0;
            result_parts.mantissa = 0;
            result_parts.hidden_bit = 1;
        }
    }
    else {
        // --- HANDLE NORMAL/DENORMAL NUMBERS ---
        if (!input_parts.sign) {
            // Case: Positive inputs (x > 0)
            // Requirement: Always return 1.0
            result_parts.exponent = 0;
            result_parts.mantissa = 0;
            result_parts.hidden_bit = 1;
        } 
        else {
            // Case: Negative inputs (x < 0)
            int32_t x_exp = input_parts.exponent;

            if (x_exp < -9) {
                // Exponent < -9 -> Return 1.0
                result_parts.exponent = 0;
                result_parts.mantissa = 0;
                result_parts.hidden_bit = 1;
            } 
            else if (x_exp > 7) {
                // Exponent > 7 -> Return +0.0
                result_parts.status.is_zero = true;
            } 
            else {
                // Range exponent -> [-9, 7] -> Use Core Approximation
                result_parts = bf16_exp2_core_approx(input_parts);
            }
        }
    }

    // --- LOGIC END ---

    // 3. Recompose result
    uint32_t result_32 = fp_recompose(result_parts, FPType::BF16);
    return static_cast<uint16_t>(result_32);
}

#endif // BF16_EXP2_HPP
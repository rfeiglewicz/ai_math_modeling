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
inline uint16_t bf16_exp2_approx(uint16_t raw_input, bool base2 = true) {
    // 1. Decompose input
    FPRaw input_parts = fp_decompose(static_cast<uint32_t>(raw_input), FPType::BF16);

    // Prepare result structure
    FPRaw result_parts = {};
    result_parts.sign = false; // 2^x is always positive (unless NaN)

    // Temporary variable for core approximation result
    FPRaw core_approx_result = {};

    // Flags for special cases
    bool set_plus_one = false;
    bool set_plus_zero = false;
    bool set_qnan_indefinite = false;

    // --- LOGIC START ---

    if (input_parts.status.is_nan) {
        // NaN -> NaN
        set_qnan_indefinite = true;
    }
    else if (input_parts.status.is_zero) {
        // 2^0 = 1
        set_plus_one = true;
    }
    else if (input_parts.status.is_inf) {
        if (input_parts.sign) {
            // 2^(-inf) = 0
            set_plus_zero = true;
        } else {
            // 2^(+inf) -> 1.0 (per requirement "positive values always return 1")
            set_plus_one = true;
        }
    }
    else {
        // --- HANDLE NORMAL/DENORMAL NUMBERS ---
        if (!input_parts.sign) {
            // Case: Positive inputs (x > 0)
            // Requirement: Always return 1.0
            set_plus_one = true;
        } 
        else {
            // Case: Negative inputs (x < 0)
            int32_t x_exp = input_parts.exponent;

            if (x_exp < -9) {
                // Exponent < -9 -> Return 1.0
                set_plus_one = true;
            } 
            else if (x_exp > 7) {
                // Exponent > 7 -> Return +0.0
                set_plus_zero = true;
            } 
            else {
                // Exponent range -> [-9, 7] -> Use Core Approximation
                core_approx_result = bf16_exp2_core_approx(input_parts, base2);
            }
        }
    }

    // --- LOGIC END ---

    // Set result based on flags
    if (set_qnan_indefinite) {
        result_parts.status.is_nan = true;
        result_parts.sign = true; // qNaN indefinite is always negative
        // Quiet NaN with cleared payload: MSB=1, rest=0
        // BF16 has 7 mantissa bits, so bit 6 is the MSB (0-indexed)
        result_parts.mantissa = 1 << 6; 
    } else if (set_plus_one) {
        result_parts.exponent = 0;
        result_parts.mantissa = 0;
        result_parts.hidden_bit = 1;
        result_parts.sign = false;
    } else if (set_plus_zero) {
        result_parts.status.is_zero = true;
        result_parts.sign = false;
    } else {
        result_parts = core_approx_result;
    }

    // 3. Recompose result
    uint32_t result_32 = fp_recompose(result_parts, FPType::BF16);
    return static_cast<uint16_t>(result_32);
}

#endif // BF16_EXP2_HPP
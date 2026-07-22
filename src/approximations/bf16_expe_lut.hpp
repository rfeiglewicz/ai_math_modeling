#ifndef BF16_EXPE_LUT_HPP
#define BF16_EXPE_LUT_HPP

#include "../utils/fp_utils.hpp"
#include "bf16_expe_lut_table.hpp"
#include <cstdint>

/**
 * @brief Full lookup-table approximation of exp(x) (base-e only) for BF16.
 *
 * Based on bf16_exp2.hpp, but the linear-approximation core (log2(e) multiply +
 * piecewise linear evaluation) is replaced by a FULL correctly-rounded lookup
 * table for every input whose unbiased exponent lies in [-9, 7].
 *
 * Logic:
 * 1. Special Cases:
 *    - NaN   -> qNaN
 *    - +/-0  -> 1.0
 *    - +Inf  -> 1.0   (per original convention: positive values return 1.0)
 *    - -Inf  -> +0.0
 * 2. Positive Inputs (x > 0):
 *    - Always Return 1.0
 * 3. Negative Inputs (x < 0):
 *    - Exp < -9      -> 1.0
 *    - Exp > 7       -> +0.0
 *    - Exp in [-9,7] -> Direct table lookup (correctly-rounded exp(x))
 *
 * The table stores RNE-rounded results, so the output is <= 0.5 ULP by
 * construction for the tabulated range.
 *
 * @param raw_input Raw 16-bit BF16 payload
 * @return Raw 16-bit BF16 result of exp(x)
 */
inline uint16_t bf16_expe_lut_approx(uint16_t raw_input) {
    // 1. Decompose input
    FPRaw input_parts = fp_decompose(static_cast<uint32_t>(raw_input), FPType::BF16);

    // Prepare result structure
    FPRaw result_parts = {};
    result_parts.sign = false; // exp(x) is always positive (unless NaN)

    // Flags for special cases
    bool set_plus_one        = false;
    bool set_plus_zero       = false;
    bool set_qnan_indefinite = false;

    // LUT path
    bool     use_lut = false;
    uint16_t lut_out = 0;

    // --- LOGIC START ---

    if (input_parts.status.is_nan) {
        // NaN -> NaN
        set_qnan_indefinite = true;
    }
    else if (input_parts.status.is_zero) {
        // e^0 = 1
        set_plus_one = true;
    }
    else if (input_parts.status.is_inf) {
        if (input_parts.sign) {
            // e^(-inf) = 0
            set_plus_zero = true;
        } else {
            // e^(+inf) -> 1.0 (per requirement "positive values always return 1")
            set_plus_one = true;
        }
    }
    else {
        // --- HANDLE NORMAL/DENORMAL NUMBERS ---
        if (!input_parts.sign) {
            // Case: Positive inputs (x > 0) -> always return 1.0
            set_plus_one = true;
        }
        else {
            // Case: Negative inputs (x < 0)
            int32_t x_exp = input_parts.exponent;

            if (x_exp < bf16_expe_lut::LUT_MIN_EXP) {
                // Exponent < -9 -> Return 1.0
                set_plus_one = true;
            }
            else if (x_exp > bf16_expe_lut::LUT_MAX_EXP) {
                // Exponent > 7 -> Return +0.0
                set_plus_zero = true;
            }
            else {
                // Exponent in [-9, 7] -> Full table lookup
                int idx = (x_exp - bf16_expe_lut::LUT_MIN_EXP) * bf16_expe_lut::LUT_MANT_COUNT
                          + static_cast<int>(input_parts.mantissa);
                lut_out = bf16_expe_lut::table[idx];
                use_lut = true;
            }
        }
    }

    // --- LOGIC END ---

    // LUT lookup returns a complete BF16 payload (sign already 0).
    if (use_lut) {
        return lut_out;
    }

    // Set result based on flags
    if (set_qnan_indefinite) {
        result_parts.status.is_nan = true;
        result_parts.sign = true; // qNaN indefinite is always negative
        result_parts.mantissa = 1 << 6; // MSB of BF16 mantissa
    } else if (set_plus_one) {
        result_parts.exponent = 0;
        result_parts.mantissa = 0;
        result_parts.hidden_bit = 1;
        result_parts.sign = false;
    } else if (set_plus_zero) {
        result_parts.status.is_zero = true;
        result_parts.sign = false;
    }

    // 3. Recompose result
    uint32_t result_32 = fp_recompose(result_parts, FPType::BF16);
    return static_cast<uint16_t>(result_32);
}

#endif // BF16_EXPE_LUT_HPP

#ifndef BF16_EXPE_HYBRID_HPP
#define BF16_EXPE_HYBRID_HPP

#include "../utils/fp_utils.hpp"
#include "bf16_expe_hybrid_table.hpp"
#include <cstdint>

/**
 * Hybrid exact-table implementation of the project-specific BF16 exp(x).
 *
 * Negative normal inputs are split by unbiased exponent:
 *   [-9, -2] : sparse threshold representation (98 seven-bit thresholds),
 *   [-1,  6] : dense 1024 x 16-bit direct ROM,
 *   7        : constant +0.0.
 * All other inputs follow the same early-out convention as bf16_expe_lut.
 */
inline uint16_t bf16_expe_hybrid_approx(uint16_t raw_input) {
    const FPRaw input = fp_decompose(static_cast<uint32_t>(raw_input), FPType::BF16);

    if (input.status.is_nan) {
        return 0xFFC0;
    }
    if (input.status.is_zero) {
        return 0x3F80;
    }
    if (input.status.is_inf) {
        return input.sign ? 0x0000 : 0x3F80;
    }
    if (!input.sign) {
        return 0x3F80;
    }

    const int exponent = input.exponent;
    if (exponent < bf16_expe_hybrid::SPARSE_MIN_EXP) {
        return 0x3F80;
    }
    if (exponent > bf16_expe_hybrid::DENSE_MAX_EXP) {
        return 0x0000;
    }

    if (exponent <= bf16_expe_hybrid::SPARSE_MAX_EXP) {
        const int exp_index = exponent - bf16_expe_hybrid::SPARSE_MIN_EXP;
        const int begin = bf16_expe_hybrid::sparse_offset[exp_index];
        const int end = bf16_expe_hybrid::sparse_offset[exp_index + 1];
        uint16_t output = bf16_expe_hybrid::sparse_base[exp_index];

        for (int i = begin; i < end; ++i) {
            if (input.mantissa >= bf16_expe_hybrid::sparse_threshold[i]) {
                --output;
            }
        }
        return output;
    }

    const int dense_index =
        (exponent - bf16_expe_hybrid::DENSE_MIN_EXP) * bf16_expe_hybrid::MANT_COUNT
        + static_cast<int>(input.mantissa);
    return bf16_expe_hybrid::dense_table[dense_index];
}

#endif

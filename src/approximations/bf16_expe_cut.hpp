#ifndef BF16_EXPE_CUT_HPP
#define BF16_EXPE_CUT_HPP

#include "../utils/fp_utils.hpp"
#include "bf16_expe_cut_table.hpp"
#include <cstdint>

/**
 * @brief Cut-point ladder implementation of the project-specific BF16 exp(x).
 *
 * Same external contract as bf16_expe_lut / bf16_expe_hybrid, but the tables no
 * longer grow with the input exponent.
 *
 * Idea
 * ----
 * Write   exp(x) = 2^(-t),  t = -x * log2(e),  I = floor(t),  f = frac(t):
 *
 *     exp(x) = 2^(-I) * 2^(-f)
 *
 * The output exponent is -I-1 (pure integer arithmetic) and the output mantissa
 * depends ONLY on f.  The (e, mant) product collapses to a single shared table:
 * all 17 input exponent bins reuse the same ladder.
 *
 * The mantissa as a function of f is a monotone step function with 128 steps,
 * so it is evaluated by cut points rather than by a 2^F entry ROM.  The stored
 * cut points are deliberately NOT the exact breakpoints
 *     thr[k] = 1 - log2((127.5 + k)/128)
 * they only have to separate the 1967 f values that can actually occur.  That
 * slack lets each cut be slid so the deviation from a linear ladder fits in
 * CUT_DEV_BITS instead of FRAC_BITS.
 *
 * A 2^L candidate ROM addressed by the top L bits of f narrows the mantissa to
 * two adjacent values, so exactly ONE comparison against ONE cut is needed --
 * no 128-wide comparator array.
 *
 * Datapath
 * --------
 *   prod = (128 + mant) * LOG2E_Q            8 x 24 multiply by a CONSTANT
 *   V    = prod >> (LOG2E_FRAC_BITS + 7 - exp - FRAC_BITS)     one barrel shift
 *   I    = V >> FRAC_BITS                    output exponent = -I-1
 *   f    = V & (2^FRAC_BITS - 1)
 *   base = lin_base(f >> (FRAC_BITS - L)) + cand_dev[]         candidate ROM
 *   mant = base + (f <= cut(base + 1))       single comparison
 *   mant == 128 -> mantissa overflow to 2.0, so mant = 0 and exponent += 1
 *
 * Special cases follow the same convention as bf16_expe_lut:
 *   NaN -> qNaN, +/-0 -> 1.0, +Inf -> 1.0, -Inf -> +0.0, x > 0 -> 1.0.
 *
 * @param raw_input Raw 16-bit BF16 payload
 * @return Raw 16-bit BF16 result of exp(x)
 */

namespace bf16_expe_cut {

/// Cut point k (1..128) reconstructed from the linear ladder plus deviation.
/// k == 129 is a sentinel that no f can ever fall below.
inline int32_t cut_at(int k) {
    if (k > CUT_COUNT) {
        return -1;
    }
    const int32_t linear = static_cast<int32_t>(128 - k) << (FRAC_BITS - 7);
    return linear + CUT_DEV_BIAS + static_cast<int32_t>(cut_dev[k - 1]);
}

/// Candidate mantissa for a bucket of the top CAND_INDEX_BITS bits of f.
inline int cand_at(int index) {
    const int linear = 128 - (index >> (CAND_INDEX_BITS - 7));
    return linear + CAND_DEV_BIAS + static_cast<int>(cand_dev[index]);
}

} // namespace bf16_expe_cut

inline uint16_t bf16_expe_cut_approx(uint16_t raw_input) {
    const FPRaw input = fp_decompose(static_cast<uint32_t>(raw_input), FPType::BF16);

    // ---- special cases -------------------------------------------------
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
        return 0x3F80;  // x > 0 saturates to 1.0 by project convention
    }

    const int exponent = static_cast<int>(input.exponent);
    const int mantissa = static_cast<int>(input.mantissa);

    // ---- range early-outs ----------------------------------------------
    if (exponent < bf16_expe_cut::MIN_EXP) {
        return 0x3F80;  // |x| too small to move the result away from 1.0
    }
    if (exponent > bf16_expe_cut::MAX_EXP) {
        return 0x0000;
    }
    if (exponent > bf16_expe_cut::ZERO_EXP
        || (exponent == bf16_expe_cut::ZERO_EXP
            && mantissa >= bf16_expe_cut::ZERO_MANT_LO)) {
        return 0x0000;  // underflows past the smallest subnormal
    }

    // ---- subnormal tail patch ------------------------------------------
    if (exponent == bf16_expe_cut::TAIL_EXP
        && mantissa >= bf16_expe_cut::TAIL_MANT_LO
        && mantissa <= bf16_expe_cut::TAIL_MANT_HI) {
        return bf16_expe_cut::tail_payload[mantissa - bf16_expe_cut::TAIL_MANT_LO];
    }

    // ---- constant multiply: t = (1.mant) * log2(e) ----------------------
    const uint64_t product =
        static_cast<uint64_t>(128 + mantissa) * bf16_expe_cut::LOG2E_Q;

    // ---- single variable shift aligns the binary point ------------------
    const int shift = bf16_expe_cut::LOG2E_FRAC_BITS + 7 - exponent
                      - bf16_expe_cut::FRAC_BITS;
    const uint64_t aligned = product >> shift;

    const int integer_part = static_cast<int>(aligned >> bf16_expe_cut::FRAC_BITS);
    const int32_t fraction =
        static_cast<int32_t>(aligned & ((1u << bf16_expe_cut::FRAC_BITS) - 1));

    // ---- candidate ROM + one comparison ---------------------------------
    const int bucket = static_cast<int>(
        fraction >> (bf16_expe_cut::FRAC_BITS - bf16_expe_cut::CAND_INDEX_BITS));
    const int base = bf16_expe_cut::cand_at(bucket);
    const int count = base + ((fraction <= bf16_expe_cut::cut_at(base + 1)) ? 1 : 0);

    // ---- pack ------------------------------------------------------------
    int out_exponent = -integer_part - 1;
    int out_mantissa = count;
    if (out_mantissa == 128) {   // mantissa rounded up to 2.0
        out_mantissa = 0;
        ++out_exponent;
    }

    return static_cast<uint16_t>(((127 + out_exponent) << 7) | out_mantissa);
}

#endif

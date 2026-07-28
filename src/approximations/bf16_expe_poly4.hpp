#ifndef BF16_EXPE_POLY4_HPP
#define BF16_EXPE_POLY4_HPP

#include "../utils/fp_utils.hpp"
#include "bf16_expe_poly4_table.hpp"
#include <cstdint>

/**
 * @brief Degree-4 minimax/Horner implementation of the project-specific
 *        BF16 exp(x), x <= 0.
 *
 * Same external contract as bf16_expe_lut / bf16_expe_hybrid / bf16_expe_cut,
 * but the back end trades table storage for DSP slices instead of the other
 * way round.  Use this variant when an FPGA has spare DSP columns and scarce
 * distributed RAM; use bf16_expe_cut when the opposite is true.
 *
 * Idea
 * ----
 * Write   exp(x) = 2^(-t),  t = -x * log2(e),  I = floor(t),  f = frac(t):
 *
 *     exp(x) = 2^(-I-1) * 2^(1-f)
 *
 * The output exponent is -I-1 (pure integer arithmetic) and the output mantissa
 * depends ONLY on f, exactly as in the cut-point ladder.  The difference is how
 * the mantissa is produced: instead of classifying f against stored cut points,
 * a degree-4 polynomial approximates 2^(1-f) and the result is rounded.
 *
 * The coefficients are NOT the minimax coefficients.  Minimax minimizes
 * |P(f) - 2^(1-f)|, but correctness only requires
 *
 *     round(128 * (P(f) - 1)) == correctly rounded mantissa code
 *
 * for the 1959 truncated f values that can actually occur.  The generator
 * therefore maximizes the worst-case distance to a rounding boundary on the
 * exact integer datapath, which absorbs coefficient quantization and per-step
 * truncation into the objective rather than adding them on top of it.  That is
 * what lets degree 4 succeed with 16-bit coefficients; a plain degree-4 minimax
 * fit misses two inputs.
 *
 * Datapath
 * --------
 *   prod = (128 + mant) * LOG2E_Q            8 x 24 multiply by a CONSTANT
 *   V    = prod >> (LOG2E_FRAC_BITS + 7 - exp - FRAC_BITS)     one barrel shift
 *   I    = V >> FRAC_BITS                    output exponent = -I-1
 *   f    = V & (2^FRAC_BITS - 1)
 *   acc  = c4
 *   acc  = (acc*f + (c_k << (ACC_FRAC - COEF_FRAC + FRAC_BITS)) + STEP_ROUND)
 *          >> FRAC_BITS                      for k = 3, 2, 1, 0   -> 4 DSPs
 *   mant = (acc - ONE + HALF) >> ROUND_SHIFT
 *   mant == 128 -> mantissa overflow to 2.0, so mant = 0 and exponent += 1
 *
 * Every Horner step is one DSP48E1: A = acc (25 b signed), B = f (17 b), and
 * C carries the coefficient with the rounding constant folded in for free.
 *
 * Special cases follow the same convention as bf16_expe_lut:
 *   NaN -> qNaN, +/-0 -> 1.0, +Inf -> 1.0, -Inf -> +0.0, x > 0 -> 1.0.
 *
 * @param raw_input Raw 16-bit BF16 payload
 * @return Raw 16-bit BF16 result of exp(x)
 */

namespace bf16_expe_poly4 {

constexpr int64_t ONE = int64_t{1} << ACC_FRAC;
constexpr int64_t HALF = int64_t{1} << (ACC_FRAC - 8);
constexpr int64_t STEP_ROUND = int64_t{1} << (FRAC_BITS - 1);

/// Integer Horner evaluation of P(f) ~= 2^(1-f), scaled by 2^ACC_FRAC.
/// Bit-for-bit identical to the RTL: one truncating shift per step, with
/// round-to-nearest folded into the additive constant.
inline int64_t horner(uint32_t fraction) {
    constexpr int shift_up = ACC_FRAC - COEF_FRAC;
    int64_t acc = static_cast<int64_t>(coef[DEGREE]) << shift_up;
    for (int k = DEGREE - 1; k >= 0; --k) {
        const int64_t c_term = (static_cast<int64_t>(coef[k]) << shift_up)
                               << FRAC_BITS;
        acc = (acc * static_cast<int64_t>(fraction) + c_term + STEP_ROUND)
              >> FRAC_BITS;
    }
    return acc;
}

} // namespace bf16_expe_poly4

inline uint16_t bf16_expe_poly4_approx(uint16_t raw_input) {
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
    if (exponent < bf16_expe_poly4::MIN_EXP) {
        return 0x3F80;  // |x| too small to move the result away from 1.0
    }
    if (exponent > bf16_expe_poly4::MAX_EXP) {
        return 0x0000;
    }
    if (exponent > bf16_expe_poly4::ZERO_EXP
        || (exponent == bf16_expe_poly4::ZERO_EXP
            && mantissa >= bf16_expe_poly4::ZERO_MANT_LO)) {
        return 0x0000;  // underflows past the smallest subnormal
    }

    // ---- subnormal tail patch ------------------------------------------
    if (exponent == bf16_expe_poly4::TAIL_EXP
        && mantissa >= bf16_expe_poly4::TAIL_MANT_LO
        && mantissa <= bf16_expe_poly4::TAIL_MANT_HI) {
        return bf16_expe_poly4::tail_payload[mantissa
                                             - bf16_expe_poly4::TAIL_MANT_LO];
    }

    // ---- constant multiply: t = (1.mant) * log2(e) ----------------------
    const uint64_t product =
        static_cast<uint64_t>(128 + mantissa) * bf16_expe_poly4::LOG2E_Q;

    // ---- single variable shift aligns the binary point ------------------
    const int shift = bf16_expe_poly4::LOG2E_FRAC_BITS + 7 - exponent
                      - bf16_expe_poly4::FRAC_BITS;
    const uint64_t aligned = product >> shift;

    const int integer_part =
        static_cast<int>(aligned >> bf16_expe_poly4::FRAC_BITS);
    const uint32_t fraction = static_cast<uint32_t>(
        aligned & ((1u << bf16_expe_poly4::FRAC_BITS) - 1));

    // ---- degree-4 Horner, then round to 7 mantissa bits -----------------
    const int64_t acc = bf16_expe_poly4::horner(fraction);
    int count = static_cast<int>((acc - bf16_expe_poly4::ONE
                                  + bf16_expe_poly4::HALF)
                                 >> bf16_expe_poly4::ROUND_SHIFT);

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

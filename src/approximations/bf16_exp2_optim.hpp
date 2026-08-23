#ifndef BF16_EXP2_OPTIM_HPP
#define BF16_EXP2_OPTIM_HPP

/**
 * @file bf16_exp2_optim.hpp
 * @brief Width-optimised, self-contained BF16 exp2 / expe core.
 *
 * Same algorithm as bf16_exp2.hpp, same results, narrower datapath. Every
 * width here was found by tests/exp_pwl_optim.cpp and is the smallest value
 * that keeps the core bit-identical to bf16_exp2_approx<29> over all 65536
 * BF16 patterns in both modes. One bit less on any of them breaks it; the
 * study prints the failing neighbours for each.
 *
 * This header is deliberately independent: it does not include
 * bf16_exp2.hpp or bf16_exp2_core.hpp, and it does not reuse bf16_cfg. The
 * production constants are all anchored to MANT_MULT_F = 29, so sharing them
 * would either drag the wide widths back in or require editing production
 * code. tests/bf16_exp2_optim_test.cpp checks the two implementations agree.
 *
 * @section widths Datapath
 *
 * | stage                | production | here      | note                     |
 * |----------------------|------------|-----------|--------------------------|
 * | mantissa source      | Q1.7   8b  | Q1.7   8b | unchanged                |
 * | log2(e) constant     | Q1.22 23b  | Q1.22 23b | unchanged                |
 * | log2(e) product      | Q2.29 31b  | Q2.21 23b | RNE, base-e path only    |
 * | unified shift reg    | Q9.38 47b  | Q9.21 30b |                          |
 * | polynomial input x   | Q0.38 38b  | Q0.17 17b | truncated slice          |
 * | coefficient a        | Q1.20 21b  | Q0.17 17b | integer bit is dead      |
 * | coefficient b        | Q1.20 21b  | Q0.18 18b | integer bit is dead      |
 * | a * x                | Q2.58 60b  | Q0.34 34b |                          |
 * | b - a*x              | Q4.58 62b  | Q0.18 18b | unsigned, no guard bits  |
 * | normalised mantissa  | Q1.58 59b  | Q1.18 19b |                          |
 * | coefficient ROM      | 128 x 42b  | 128 x 35b |                          |
 *
 * @section elims Structural eliminations
 *
 * Three pieces of production hardware are not merely narrower here, they are
 * gone. All three follow from one measured fact: b - a*x lands in [0.5, 1)
 * for every reachable input, because max(a*x) = 0.347323 and min(b) = 0.847980.
 *
 *  - The sign bit and the four integer guard bits of the accumulator. The
 *    subtraction cannot go negative and cannot exceed 1, so it is an
 *    unsigned Q0.18 operation.
 *  - The priority encoder over 62 bits.
 *  - The normalising barrel shifter over 62 bits.
 *
 * @section norm Normalisation, and one honest caveat
 *
 * "b - a*x is in [0.5, 1)" holds for every x the front end can produce, but
 * not for every x the datatype can hold. The linear fit of the last segment
 * undershoots by one accumulator LSB at x = 131071, the single largest code.
 * The largest reachable x is 130823, so nothing hits it, but the statement is
 * an input-space property rather than an identity.
 *
 * Rather than lean on that, the default here keeps a guard, because it turns
 * out to cost nothing:
 *
 *   - static (STATIC_NORM = true): shift left by 1, exponent -1. Assumes the
 *     input-space property.
 *   - guarded (STATIC_NORM = false, default): test bit CALC_F-1 and shift by
 *     1 or 2, exponent -1 or -2.
 *
 * Both produce a 19-bit Q1.18 mantissa in [1.0, 2.0): res in [0.5, 1) shifted
 * by 1 and res in [0.25, 0.5) shifted by 2 land in the same range, so the
 * guarded form is not one bit wider. It is one bit test and a 2:1 mux against
 * a 62-bit priority encoder and a 62-bit barrel shifter, and it removes the
 * obligation to re-prove reachability whenever the table, MANT_MULT_F, X_F or
 * the input exponent range changes. The two agree bit-for-bit on every
 * reachable input; tests/bf16_exp2_optim_test.cpp checks that.
 *
 * The guarded form still assumes b - a*x >= 0.25, which has 1 ULP of margin
 * at the accumulator's own resolution and is therefore not a close call.
 *
 * @section subtract The subtraction
 *
 * a*x carries 34 fractional bits and b carries 18, so the naive form aligns
 * both to Q0.34 and truncates the difference back to Q0.18. That is a 34-bit
 * subtractor whose low 16 output bits are discarded.
 *
 * Since the low 16 bits of the aligned b are zero, the same result comes out
 * of an 18-bit subtractor:
 *
 *     floor(((b << 16) - ax) / 2^16) = b - ceil(ax / 2^16)
 *                                    = b - (ax >> 16) - (ax[15:0] != 0)
 *
 * i.e. an 18-bit subtract, a 16-input OR reduce and a borrow-in. That is the
 * form implemented below, so the model matches what the RTL should build.
 *
 * @section rtl Mapping to RTL parameters
 *
 *   MANT_MULT_F  -> bf16_log2e_mult    MANT_MULT_ROUND_FRAC
 *   UNIFIED_F    -> bf16_unified_shift IN_F
 *   X_F          -> bf16_linear_approx MULT_X_W
 *   A_F / B_F    -> bf16_coeff_rom     COEFF_F, split per coefficient
 *   CALC_F       -> bf16_linear_approx CALC_F
 *   static norm  -> bf16_normalize replaced by a wire
 */

#include "../utils/fp_utils.hpp"
#include "../../modeling/coeff_gen/bf16_exp2_optim_coeffs.hpp"
#include "ac_int.h"
#include "ac_fixed.h"

/**
 * @namespace bf16_optim_cfg
 * @brief Fixed-point widths for the optimised core.
 *
 * Self-contained on purpose: nothing here refers to bf16_cfg.
 */
namespace bf16_optim_cfg {

    // --- BF16 target format ------------------------------------------------
    constexpr int TARGET_MANT_W   = 7;
    constexpr int TARGET_EXP_BIAS = 127;
    constexpr int TARGET_MIN_EXP  = 1 - TARGET_EXP_BIAS;

    /** @brief Extended mantissa for rounding: carry + hidden + mantissa. */
    constexpr int EXT_MANT_W    = TARGET_MANT_W + 2;
    constexpr int CARRY_BIT_IDX = EXT_MANT_W - 1;
    constexpr int HIDDEN_BIT_IDX = TARGET_MANT_W;

    /** @brief Input exponent range handled by the polynomial core. */
    constexpr int INPUT_MIN_EXP = -9;
    constexpr int INPUT_MAX_EXP = 7;

    // --- Mantissa preparation ---------------------------------------------
    /** @brief Mantissa with hidden bit: always in [1.0, 2.0). */
    constexpr int MANT_SRC_I = 1;
    constexpr int MANT_SRC_F = TARGET_MANT_W;
    constexpr int MANT_SRC_W = MANT_SRC_I + MANT_SRC_F;

    // --- Base conversion ---------------------------------------------------
    constexpr int LOG2E_I = bf16_exp2_optim_packed::LOG2E_I;
    constexpr int LOG2E_F = bf16_exp2_optim_packed::LOG2E_F;
    constexpr int LOG2E_W = bf16_exp2_optim_packed::LOG2E_W;

    /** @brief Raw product mant_src * log2(e), before rounding. */
    constexpr int MANT_MULT_I      = MANT_SRC_I + LOG2E_I;
    constexpr int MANT_MULT_FULL_F = MANT_SRC_F + LOG2E_F;
    constexpr int MANT_MULT_FULL_W = MANT_MULT_I + MANT_MULT_FULL_F;

    /**
     * @brief Fractional bits kept after RNE rounding of that product.
     *
     * 21 is the minimum. 22 is the minimum if the rounder is replaced by plain
     * truncation; round-half-up also reaches 21 and is cheaper than RNE in
     * hardware, since the constant can ride in on a DSP48 C port.
     */
    constexpr int MANT_MULT_F = 21;
    constexpr int MANT_MULT_W = MANT_MULT_I + MANT_MULT_F;

    // --- Unified shift register -------------------------------------------
    /** @brief Integer bits needed for the largest left shift. */
    constexpr int UNIFIED_I = INPUT_MAX_EXP + MANT_MULT_I;
    /**
     * @brief Fractional bits of the shift register.
     *
     * Cannot go below MANT_MULT_F: bits present in the product would be lost
     * before the right shift can move them into range.
     */
    constexpr int UNIFIED_F = 21;
    constexpr int UNIFIED_W = UNIFIED_I + UNIFIED_F;

    // --- Polynomial input --------------------------------------------------
    /** @brief Fractional bits of x entering the multiplier (truncated slice). */
    constexpr int X_F = 17;
    constexpr int X_W = X_F;

    // --- Look-up table -----------------------------------------------------
    constexpr int LUT_SIZE    = bf16_exp2_optim_packed::LUT_SIZE;
    constexpr int LUT_ADDR_W  = 7;
    constexpr int LUT_MAX_IDX = LUT_SIZE - 1;
    constexpr int PACKED_W    = bf16_exp2_optim_packed::PACKED_W;

    static_assert(1 << LUT_ADDR_W == LUT_SIZE, "LUT_ADDR_W must address LUT_SIZE");

    // --- Coefficients ------------------------------------------------------
    /** @brief Slope: unsigned Q0.17, no integer bit. */
    constexpr int A_F = bf16_exp2_optim_packed::A_F;
    constexpr int A_W = bf16_exp2_optim_packed::A_W;
    /** @brief Offset: unsigned Q0.18, no integer bit. */
    constexpr int B_F = bf16_exp2_optim_packed::B_F;
    constexpr int B_W = bf16_exp2_optim_packed::B_W;

    // --- Product and accumulator -------------------------------------------
    /** @brief a * x: both operands are < 1, so the product is too. */
    constexpr int PROD_F = A_F + X_F;
    constexpr int PROD_W = PROD_F;

    /** @brief b - a*x: unsigned, always in [0.5, 1). */
    constexpr int CALC_F = 18;
    constexpr int CALC_W = CALC_F;

    /** @brief Bits of a*x dropped when aligning to the accumulator. */
    constexpr int CDROP = PROD_F - CALC_F;

    static_assert(CDROP >= 0, "a*x must be at least as fine as the accumulator");
    static_assert(B_F == CALC_F, "b feeds the accumulator directly");

    // --- Normalisation and rounding ----------------------------------------
    /** @brief MSB position of b - a*x for every reachable input. */
    constexpr int STATIC_MSB_IDX = CALC_F - 1;
    /** @brief Normalisation exponent that goes with it, STATIC_MSB_IDX - CALC_F. */
    constexpr int STATIC_POLY_EXP = -1;

    /**
     * @brief Normalised mantissa: Q1.18, always in [1.0, 2.0).
     *
     * Width is the same for both normalisation variants. res in [0.5, 1)
     * shifted left by 1 and res in [0.25, 0.5) shifted left by 2 both land in
     * [2^18, 2^19), so guarding the shift costs no extra bit.
     */
    constexpr int POLY_OUT_I = 1;
    constexpr int POLY_OUT_F = CALC_F;
    constexpr int POLY_OUT_W = POLY_OUT_I + POLY_OUT_F;

    /** @brief Bits discarded to the right by the final round-to-BF16. */
    constexpr int BASE_SHIFT = POLY_OUT_F - TARGET_MANT_W;
}

/** @brief Polynomial input: unsigned Q0.17. */
typedef ac_int<bf16_optim_cfg::X_W, false> optim_x_t;

/** @brief Unified shift register: unsigned Q9.21. */
typedef ac_fixed<bf16_optim_cfg::UNIFIED_W, bf16_optim_cfg::UNIFIED_I, false> optim_unified_t;

/** @brief Normalised polynomial result. */
struct OptimPolyResult {
    ac_int<bf16_optim_cfg::POLY_OUT_W, false> mantissa;  ///< Q1.18, in [1.0, 2.0)
    int32_t exponent;                                    ///< always STATIC_POLY_EXP
};

/**
 * @brief Evaluates 2^(-x) for x in [0, 1) by piecewise linear interpolation.
 *
 * Segment index comes from the top LUT_ADDR_W bits of @p x, inverted to map
 * the decreasing 2^-x onto an increasing table.
 *
 * @tparam STATIC_NORM  true  -> assume b - a*x >= 0.5 and shift by a constant 1.
 *                      false -> test the top bit and shift by 1 or 2 (default).
 *                      Identical results on every reachable input; see the
 *                      normalisation section of the file header.
 * @param x Polynomial input, unsigned Q0.17.
 * @return Normalised mantissa and its exponent.
 */
template<bool STATIC_NORM = false>
inline OptimPolyResult bf16_exp2_optim_poly(optim_x_t x) {
    namespace cfg = bf16_optim_cfg;

    // --- Segment lookup ----------------------------------------------------
    const int raw_index = x.template slc<cfg::LUT_ADDR_W>(cfg::X_F - cfg::LUT_ADDR_W).to_int();
    const int idx = cfg::LUT_MAX_IDX - raw_index;

    const uint64_t packed = bf16_exp2_optim_packed::coeffs[idx];
    const ac_int<cfg::A_W, false> a = static_cast<uint32_t>(packed & ((1ull << cfg::A_W) - 1));
    const ac_int<cfg::B_W, false> b = static_cast<uint32_t>(packed >> cfg::A_W);

    // --- a * x, unsigned Q0.34 --------------------------------------------
    const ac_int<cfg::PROD_W, false> ax = a * x;

    // --- b - a*x, narrow form ---------------------------------------------
    // Truncating the aligned 34-bit difference down to Q0.18 is the same as
    // subtracting the rounded-up high part of a*x. See the header comment.
    const ac_int<cfg::CALC_W, false> ax_hi = ax.template slc<cfg::CALC_W>(cfg::CDROP);
    const bool ax_lo_nonzero = (ax.template slc<cfg::CDROP>(0) != 0);

    const ac_int<cfg::CALC_W, false> borrow = ax_lo_nonzero ? 1 : 0;
    const ac_int<cfg::CALC_W, false> res = b - ax_hi - borrow;

    // --- Normalisation: no priority encoder, no barrel shifter -------------
    OptimPolyResult result;

    if (STATIC_NORM) {
        // res >= 0.5 for every reachable x, so the shift is a constant.
        result.exponent = cfg::STATIC_POLY_EXP;
        result.mantissa = ((ac_int<cfg::POLY_OUT_W, false>)res) << 1;
    } else {
        // One bit test and a 2:1 mux. Covers res down to 0.25, which is far
        // below anything the fit produces, at no extra width.
        const bool top = res[cfg::CALC_F - 1];
        result.exponent = top ? cfg::STATIC_POLY_EXP : (cfg::STATIC_POLY_EXP - 1);
        result.mantissa = ((ac_int<cfg::POLY_OUT_W, false>)res) << (top ? 1 : 2);
    }

    return result;
}

/**
 * @brief Checks that @p r came out normalised, i.e. its MSB is set.
 *
 * With the guarded normalisation this is true by construction as long as
 * b - a*x >= 0.25. With STATIC_NORM it is the input-space assumption itself,
 * and tests/bf16_exp2_optim_test.cpp enumerates the reachable set to enforce
 * it.
 */
inline bool bf16_exp2_optim_static_norm_holds(const OptimPolyResult& r) {
    return r.mantissa[bf16_optim_cfg::POLY_OUT_W - 1];
}

/** @brief Range-reduction output: polynomial input and exponent adjustment. */
struct OptimFrontEnd {
    optim_x_t x;               ///< polynomial input, unsigned Q0.17
    int32_t   exponent_bias;   ///< negated integer part of the shifted value
};

/**
 * @brief Mantissa preparation, base conversion and range reduction to [0, 1).
 *
 * Split out of the core so that tests can enumerate exactly which polynomial
 * inputs are reachable, which is what justifies the static normalisation.
 *
 * @param input_parts Decomposed BF16 input, exponent in
 *                    [INPUT_MIN_EXP, INPUT_MAX_EXP].
 * @param base2       true -> 2^x, false -> e^x.
 */
inline OptimFrontEnd bf16_exp2_optim_front_end(const FPRaw& input_parts, bool base2) {
    namespace cfg = bf16_optim_cfg;

    // --- 1. Mantissa with hidden bit: Q1.7, in [1.0, 2.0) ------------------
    ac_fixed<cfg::MANT_SRC_W, cfg::MANT_SRC_I, false> mant_src;
    mant_src[cfg::MANT_SRC_W - 1] = 1;
    mant_src.set_slc(0, (ac_int<cfg::TARGET_MANT_W, false>)input_parts.mantissa);

    // --- 2. Base conversion: mant_src * log2(e), rounded to Q2.21 ----------
    // Base-2 bypasses this entirely, so the multiplier and its rounder sit on
    // the base-e path only.
    ac_fixed<cfg::LOG2E_W, cfg::LOG2E_I, false> log2e_const;
    log2e_const.set_slc(0, ac_int<cfg::LOG2E_W, false>(bf16_exp2_optim_packed::log2e_int_val));

    ac_fixed<cfg::MANT_MULT_FULL_W, cfg::MANT_MULT_I, false> mant_mult = mant_src * log2e_const;
    ac_fixed<cfg::MANT_MULT_W, cfg::MANT_MULT_I, false, AC_RND_CONV> mant_mult_rnd = mant_mult;

    // --- 3. Unified format and exponent shift ------------------------------
    optim_unified_t val = base2 ? (optim_unified_t)mant_src : (optim_unified_t)mant_mult_rnd;

    const int32_t e = input_parts.exponent;
    if (e >= 0) val <<= e;
    else        val >>= (-e);

    // --- 4. Split: integer part sets the exponent, top X_F fraction bits
    //        become the polynomial input (plain truncation).
    OptimFrontEnd fe;
    fe.exponent_bias = -(int)val.to_int();
    fe.x = val.template slc<cfg::X_F>(cfg::UNIFIED_F - cfg::X_F);
    return fe;
}

/**
 * @brief Core approximation of 2^x or e^x for a decomposed, in-range input.
 *
 * Range-reduces x to [0, 1), evaluates the polynomial and rounds to BF16 with
 * round-to-nearest-even, including the subnormal path.
 *
 * @tparam STATIC_NORM  Normalisation variant, see bf16_exp2_optim_poly.
 * @param input_parts Decomposed BF16 input, sign set, exponent in
 *                    [INPUT_MIN_EXP, INPUT_MAX_EXP].
 * @param base2       true -> 2^x, false -> e^x.
 * @return Decomposed BF16 result.
 */
template<bool STATIC_NORM = false>
inline FPRaw bf16_exp2_optim_core_approx(const FPRaw& input_parts, bool base2 = true) {
    namespace cfg = bf16_optim_cfg;

    const OptimFrontEnd fe = bf16_exp2_optim_front_end(input_parts, base2);
    const OptimPolyResult poly_res = bf16_exp2_optim_poly<STATIC_NORM>(fe.x);

    // --- 5. Alignment for rounding -----------------------------------------
    const int32_t final_exponent = poly_res.exponent + fe.exponent_bias;
    const bool is_sub = (final_exponent < cfg::TARGET_MIN_EXP);
    const int shift_val = cfg::BASE_SHIFT + (is_sub ? (cfg::TARGET_MIN_EXP - final_exponent) : 0);

    const ac_int<cfg::POLY_OUT_W, false> m_raw = poly_res.mantissa;

    // --- 6. Round to nearest even ------------------------------------------
    const bool lsb_bit   = (shift_val < cfg::POLY_OUT_W) ? (bool)m_raw[shift_val] : false;
    const bool guard_bit = (shift_val > 0 && shift_val <= cfg::POLY_OUT_W)
                         ? (bool)m_raw[shift_val - 1] : false;

    bool sticky_bit = false;
    if (shift_val > 1) {
        if (shift_val > cfg::POLY_OUT_W) {
            sticky_bit = (m_raw != 0);
        } else {
            const ac_int<cfg::POLY_OUT_W, false> mask =
                (ac_int<cfg::POLY_OUT_W, false>(1) << (shift_val - 1)) - 1;
            sticky_bit = ((m_raw & mask) != 0);
        }
    }

    const bool round_up = guard_bit && (lsb_bit || sticky_bit);

    ac_int<cfg::EXT_MANT_W, false> result_m_ext = 0;
    if (shift_val < cfg::POLY_OUT_W) {
        result_m_ext = (ac_int<cfg::EXT_MANT_W, false>)(m_raw >> shift_val);
    }
    if (round_up) result_m_ext++;

    // --- 7. Post-rounding carry --------------------------------------------
    int32_t adjusted_exp = is_sub ? cfg::TARGET_MIN_EXP : final_exponent;
    if (result_m_ext[cfg::CARRY_BIT_IDX]) {
        adjusted_exp++;
        result_m_ext >>= 1;
    }

    // --- 8. Assemble --------------------------------------------------------
    FPRaw result = {};
    result.sign = 0;

    if (result_m_ext == 0) {
        result.status.is_zero = true;
        result.exponent = 0;
    } else if (is_sub && !result_m_ext[cfg::HIDDEN_BIT_IDX]) {
        result.mantissa = result_m_ext.template slc<cfg::TARGET_MANT_W>(0);
        result.hidden_bit = 0;
        result.exponent = cfg::TARGET_MIN_EXP - 1;
        result.status.is_denormal = true;
    } else {
        result.mantissa = result_m_ext.template slc<cfg::TARGET_MANT_W>(0);
        result.hidden_bit = 1;
        result.exponent = adjusted_exp;
        result.status.is_denormal = false;
    }

    return result;
}

/**
 * @brief Width-optimised BF16 exp2 / expe.
 *
 * Special cases follow bf16_exp2_approx exactly:
 *   NaN            -> qNaN
 *   +/-0           -> 1.0
 *   +Inf, x > 0    -> 1.0
 *   -Inf           -> +0.0
 *   exponent < -9  -> 1.0     (result rounds to 1.0 anyway)
 *   exponent > 7   -> +0.0    (result underflows)
 *
 * @tparam STATIC_NORM Normalisation variant, see bf16_exp2_optim_poly.
 * @param raw_input Raw 16-bit BF16 payload.
 * @param base2     true -> 2^x, false -> e^x.
 * @return Raw 16-bit BF16 result.
 */
template<bool STATIC_NORM = false>
inline uint16_t bf16_exp2_optim_approx(uint16_t raw_input, bool base2 = true) {
    namespace cfg = bf16_optim_cfg;

    const FPRaw input_parts = fp_decompose(static_cast<uint32_t>(raw_input), FPType::BF16);

    FPRaw result_parts = {};
    result_parts.sign = false;

    FPRaw core_result = {};
    bool set_plus_one = false, set_plus_zero = false, set_qnan = false;

    if (input_parts.status.is_nan) {
        set_qnan = true;
    } else if (input_parts.status.is_zero) {
        set_plus_one = true;
    } else if (input_parts.status.is_inf) {
        if (input_parts.sign) set_plus_zero = true;
        else                  set_plus_one  = true;
    } else if (!input_parts.sign) {
        set_plus_one = true;
    } else if (input_parts.exponent < cfg::INPUT_MIN_EXP) {
        set_plus_one = true;
    } else if (input_parts.exponent > cfg::INPUT_MAX_EXP) {
        set_plus_zero = true;
    } else {
        core_result = bf16_exp2_optim_core_approx<STATIC_NORM>(input_parts, base2);
    }

    if (set_qnan) {
        result_parts.status.is_nan = true;
        result_parts.sign = true;
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
        result_parts = core_result;
    }

    return static_cast<uint16_t>(fp_recompose(result_parts, FPType::BF16));
}

#endif // BF16_EXP2_OPTIM_HPP

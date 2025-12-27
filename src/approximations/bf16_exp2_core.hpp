#ifndef BF16_EXP2_CORE_HPP
#define BF16_EXP2_CORE_HPP

#include "../utils/fp_utils.hpp"
#include "../../modeling/coeff_gen/bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"

/**
 * @namespace bf16_cfg
 * @brief Configuration constants for fixed-point arithmetic and BF16 format alignment.
 * * This namespace defines the bit-widths and precision parameters used throughout the 
 * piecewise linear approximation and RNE rounding logic.
 */
namespace bf16_cfg {
    /** @brief BF16 target format parameters. */
    constexpr int TARGET_MANT_W = 7;
    constexpr int TARGET_EXP_BIAS = 127;
    constexpr int TARGET_MIN_EXP = 1 - TARGET_EXP_BIAS; // -126 for BF16

    /** @brief Input Exponent Range for Approximation [-9, 7] */
    constexpr int INPUT_MIN_EXP = -9;
    constexpr int INPUT_MAX_EXP = 7;

    /** @brief Mantissa Source Format (1.7) */
    constexpr int MANT_SRC_I = 1;
    constexpr int MANT_SRC_F = TARGET_MANT_W;
    constexpr int MANT_SRC_W = MANT_SRC_I + MANT_SRC_F;

    /** @brief Log2E Constant Format (1.25) */
    constexpr int LOG2E_I = 1;
    constexpr int LOG2E_F = 25;
    constexpr int LOG2E_W = LOG2E_I + LOG2E_F;

    /** @brief Multiplication Result Format (mant_src * log2e) -> 2.32 */
    constexpr int MANT_MULT_I = MANT_SRC_I + LOG2E_I;
    constexpr int MANT_MULT_F = MANT_SRC_F + LOG2E_F;
    constexpr int MANT_MULT_W = MANT_MULT_I + MANT_MULT_F;

    /** @brief Input format: unsigned 1.41 (1 integer bit, 41 fractional bits). 
     * Derived from multiplication precision and max negative shift.
     */
    constexpr int IN_I = 1;
    constexpr int IN_F = MANT_MULT_F + (-INPUT_MIN_EXP); // 32 + 9 = 41
    constexpr int IN_W = IN_I + IN_F;

    /** @brief Look-Up Table (LUT) parameters for piecewise segments. */
    constexpr int LUT_ADDR_W = 6;               // Number of address bits (log2 of LUT size)
    constexpr int LUT_SIZE = 1 << LUT_ADDR_W;   // LUT size (64 entries)
    constexpr int LUT_MAX_IDX = LUT_SIZE - 1;   // Maximum index for coefficient mapping

    /** @brief Coefficient format: unsigned 1.25. */
    constexpr int COEFF_I = 1;
    constexpr int COEFF_F = 25;
    constexpr int COEFF_W = COEFF_I + COEFF_F;

    /** @brief Multiplication parameters (a * x). Product width is the sum of operand widths. */
    constexpr int MULT_I = IN_I + COEFF_I; // 2 integer bits
    constexpr int MULT_F = IN_F + COEFF_F; // 41 fractional bits
    constexpr int MULT_W = MULT_I + MULT_F;

    /** @brief Sign bit requirement for intermediate signed calculations. */
    constexpr int CALC_SIGN_BIT = 1; 


    /** * @brief Intermediate calculation parameters (b - ax).
     * The CALC format is sized to accommodate the widest operand without loss of precision.
     */
    constexpr int MAX_OP_I = (MULT_I + CALC_SIGN_BIT > COEFF_I) ? MULT_I + CALC_SIGN_BIT : COEFF_I;
    constexpr int MAX_OP_F = (MULT_F > COEFF_F) ? MULT_F : COEFF_F;

    /** @brief Guard bit to prevent overflow during addition/subtraction. */
    constexpr int CALC_ADD_GUARD = 1; 
    
    /** @brief Total integer and fractional bits for the calculation pipeline. */
    constexpr int CALC_I = MAX_OP_I + CALC_ADD_GUARD; 
    constexpr int CALC_F = MAX_OP_F;
    constexpr int CALC_W = CALC_I + CALC_F;

    /** @brief Polynomial output format: 1.41 unsigned (41 bits for RNE precision). */
    constexpr int POLY_OUT_I = 1;
    constexpr int POLY_OUT_F = CALC_F;
    constexpr int POLY_OUT_W = POLY_OUT_I + POLY_OUT_F;

    /** * @brief Alignment and rounding logic parameters.
     * BASE_SHIFT is the difference between polynomial precision and target mantissa.
     */
    constexpr int BASE_SHIFT = POLY_OUT_F - TARGET_MANT_W; 
    
    /** @brief Extended mantissa: includes carry bit, hidden bit, and mantissa bits. */
    constexpr int EXT_MANT_W = TARGET_MANT_W + 2;    // 9 bits total
    constexpr int CARRY_BIT_IDX = EXT_MANT_W - 1;    // Index of the overflow bit (8)
    constexpr int HIDDEN_BIT_IDX = TARGET_MANT_W;    // Index of the hidden bit (7)

    /** @brief Input conversion parameters for decomposing x into integer and fractional parts. */
    constexpr int IN_CONV_INT_W = INPUT_MAX_EXP + MANT_MULT_I; // 7 + 2 = 9
    constexpr int IN_CONV_FRAC_W = TARGET_MANT_W;    // Bits for fractional part of x
    constexpr int IN_CONV_W = IN_CONV_INT_W + IN_CONV_FRAC_W;
}

/** @brief Typedef for input fixed-point mantissa. */
typedef ac_fixed<bf16_cfg::IN_W, bf16_cfg::IN_I, false> mant_t;

/** @brief Structure holding normalized polynomial result. */
struct PolyResult {
    ac_fixed<bf16_cfg::POLY_OUT_W, bf16_cfg::POLY_OUT_I, false> mantissa; 
    int32_t exponent;
};

/**
 * @brief Calculates 2^(-x) using piecewise linear approximation for x in [0, 1].
 * * Formula: result = a * (-x) + b
 * Uses a Look-Up Table (LUT) for coefficients 'a' and 'b' based on the leading 
 * bits of the input fractional part.
 * * @param mant_val Input value in 1.41 fixed-point format.
 * @return Normalized PolyResult containing mantissa and exponent.
 */
inline PolyResult bf16_exp2_poly(mant_t mant_val) {
    // Extract LUT index from the MSBs of the fractional part
    uint8_t lut_index = mant_val.slc<bf16_cfg::LUT_ADDR_W>(bf16_cfg::IN_F - bf16_cfg::LUT_ADDR_W);

    // Fetch coefficients based on the inverted index for the 2^-x mapping
    float a_float = bf16_exp2::coeffs_a[bf16_cfg::LUT_MAX_IDX - lut_index];
    float b_float = bf16_exp2::coeffs_b[bf16_cfg::LUT_MAX_IDX - lut_index];

    typedef ac_fixed<bf16_cfg::COEFF_W, bf16_cfg::COEFF_I, false> coeff_t;
    typedef ac_fixed<bf16_cfg::CALC_W, bf16_cfg::CALC_I, true> calc_t;

    coeff_t a_fixed = a_float;
    coeff_t b_fixed = b_float;

    // Perform multiplication: a * x
    ac_fixed<bf16_cfg::MULT_W, bf16_cfg::MULT_I, false> ax_u = a_fixed * mant_val;

    // Negate and cast to signed type: -ax
    calc_t ax_s = - (calc_t)ax_u;
    
    // Add offset: res = b + (-ax)
    calc_t res = ax_s + (calc_t)b_fixed;
    
    // Treat result as raw bits for normalization logic
    ac_int<bf16_cfg::CALC_W, false> res_raw = res.slc<bf16_cfg::CALC_W>(0);

    // 1. Priority Encoder (Find MSB)
    // Standard HLS pattern that synthesizes into fast combinational logic.
    int msb_idx = -1;
    for (int i = bf16_cfg::CALC_W - 1; i >= 0; --i) {
        if (res_raw[i]) {
            msb_idx = i;
            break;
        }
    }

    PolyResult result;
    // Calculate exponent relative to 1.41 format
    result.exponent = msb_idx - bf16_cfg::POLY_OUT_F;

    // 2. Normalization (Barrel Shifter + Slice)
    // Aligns MSB to the left (position CALC_W - 1) to maximize precision.
    int shift = (bf16_cfg::CALC_W - 1) - msb_idx;
    
    // Barrel Shifting
    ac_int<bf16_cfg::CALC_W, false> normalized = res_raw << shift;

    // Slicing: Extract POLY_OUT_W most significant bits
    result.mantissa.set_slc(0, normalized.slc<bf16_cfg::POLY_OUT_W>(bf16_cfg::CALC_W - bf16_cfg::POLY_OUT_W));
    
    return result;
}

/**
 * @brief Core hardware-accurate approximation of exp(x) (base e) or exp2(x) (base 2).
 * * Handles input decomposition, range reduction to [0, 1], polynomial evaluation,
 * and standard BF16 rounding (Round to Nearest Even).
 * * @param input_parts Decomposed BF16 input structure.
 * @param base2 If true, calculates 2^x. If false, calculates e^x.
 * @return Decomposed BF16 result structure.
 */
inline FPRaw bf16_exp2_core_approx(const FPRaw& input_parts, bool base2 = true) {
    FPRaw result = {};
    mant_t mant_val;
    int32_t temp_exponent = input_parts.exponent;
    int32_t exponent_bias = 0;

    // Convert input to 1.41 fixed-point format
    // Unified 9.41 format (9 integer bits, 41 fractional bits)
    typedef ac_fixed<bf16_cfg::IN_CONV_INT_W + bf16_cfg::IN_F, bf16_cfg::IN_CONV_INT_W, false> unified_t;
    unified_t val = 0;

    // 1. Prepare Mantissa (1.7 format)
    ac_fixed<bf16_cfg::MANT_SRC_W, bf16_cfg::MANT_SRC_I, false> mant_src;
    mant_src[bf16_cfg::MANT_SRC_W - 1] = 1; // Hidden bit
    mant_src.set_slc(0, (ac_int<bf16_cfg::TARGET_MANT_W, false>)input_parts.mantissa);

    // 2. Multiply by log2(e) (1.25 format)
    // log2(e) ~= 1.442695
    ac_fixed<bf16_cfg::LOG2E_W, bf16_cfg::LOG2E_I, false> log2e_const = 1.4426950408889634073599246810018921374266459541529859341354494069;
    
    // Result is 2.32 format (2 integer bits, 32 fractional bits)
    ac_fixed<bf16_cfg::MANT_MULT_W, bf16_cfg::MANT_MULT_I, false> mant_mult = mant_src * log2e_const;

    // 3. Move to Unified Format (9.41)
    // We need 9 integer bits because the maximum negative exponent is -9,
    // so the shift will be 9. The multiplication result has 2 integer bits.
    val = base2 ? (unified_t)mant_src : (unified_t)mant_mult;

    // Shift based on exponent
    if (temp_exponent >= 0) {
        val <<= temp_exponent;
    } else {
        val >>= (-temp_exponent);
    }

    // Extract fractional part for polynomial approximation
    mant_val = 0;
    mant_val.set_slc(0, val.slc<bf16_cfg::IN_F>(0));

    // Integer part determines the final exponent shift
    exponent_bias = -(int)val.to_int();

    // Call polynomial approximation
    PolyResult poly_res = bf16_exp2_poly(mant_val);

    int32_t final_exponent = poly_res.exponent + exponent_bias;
    ac_int<bf16_cfg::POLY_OUT_W, false> full_mant = poly_res.mantissa.slc<bf16_cfg::POLY_OUT_W>(0);
    
    // 1. Alignment Logic for Rounding
    // Check if the result falls into subnormal range for BF16
    bool is_sub = (final_exponent < bf16_cfg::TARGET_MIN_EXP);
    
    // shift_val: number of bits discarded to the right during rounding
    int shift_val = bf16_cfg::BASE_SHIFT + (is_sub ? (bf16_cfg::TARGET_MIN_EXP - final_exponent) : 0);

    // 2. Rounding Bit Extraction (RNE Logic)
    ac_int<bf16_cfg::POLY_OUT_W, false> m_raw = full_mant;
    
    bool lsb_bit = (shift_val < bf16_cfg::POLY_OUT_W) ? (bool)m_raw[shift_val] : false;
    bool guard_bit = (shift_val > 0 && shift_val <= bf16_cfg::POLY_OUT_W) ? (bool)m_raw[shift_val - 1] : false;
    
    bool sticky_bit = false;
    if (shift_val > 1) {
        if (shift_val > bf16_cfg::POLY_OUT_W) {
            sticky_bit = (m_raw != 0); 
        } else {
            ac_int<bf16_cfg::POLY_OUT_W, false> mask = (ac_int<bf16_cfg::POLY_OUT_W, false>(1) << (shift_val - 1)) - 1;
            sticky_bit = (m_raw & mask) != 0;
        }
    }

    // Determine if we should round up based on Round-to-Nearest-Even (RNE)
    bool round_up = guard_bit && (lsb_bit || sticky_bit);

    // 3. Shift and Round
    ac_int<bf16_cfg::EXT_MANT_W, false> result_m_ext = 0;
    if (shift_val < bf16_cfg::POLY_OUT_W) {
        result_m_ext = (ac_int<bf16_cfg::EXT_MANT_W, false>)(m_raw >> shift_val);
    }
    if (round_up) result_m_ext++;

    // 4. Post-rounding Normalization
    // Adjust exponent if rounding caused an overflow (carry-out bit)
    int32_t adjusted_exp = is_sub ? bf16_cfg::TARGET_MIN_EXP : final_exponent;
    if (result_m_ext[bf16_cfg::CARRY_BIT_IDX]) {
        adjusted_exp++;
        result_m_ext >>= 1;
    }

    // 5. Final BF16 Structure Formation
    result.sign = 0;
    if (result_m_ext == 0) {
        result.status.is_zero = true;
        result.exponent = 0;
    } else if (is_sub && !result_m_ext[bf16_cfg::HIDDEN_BIT_IDX]) {
        // Result is a denormal number
        result.mantissa = result_m_ext.slc<bf16_cfg::TARGET_MANT_W>(0);
        result.hidden_bit = 0;
        result.exponent = bf16_cfg::TARGET_MIN_EXP - 1; // -127 for BF16
        result.status.is_denormal = true;
    } else {
        // Result is a normal number
        result.mantissa = result_m_ext.slc<bf16_cfg::TARGET_MANT_W>(0);
        result.hidden_bit = 1;
        result.exponent = adjusted_exp;
        result.status.is_denormal = false;
    }
    
    return result;
}

#endif // BF16_EXP2_CORE_HPP
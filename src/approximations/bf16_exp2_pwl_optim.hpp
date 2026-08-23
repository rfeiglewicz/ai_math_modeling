#ifndef BF16_EXP2_PWL_OPTIM_HPP
#define BF16_EXP2_PWL_OPTIM_HPP

/**
 * @file bf16_exp2_pwl_optim.hpp
 * @brief Width-parameterised replica of bf16_exp2_core_approx used to search
 *        for the narrowest datapath that stays bit-exact with production.
 *
 * The production model (bf16_exp2_core.hpp) derives every intermediate width
 * from the compile-time constants in @ref bf16_cfg, all of which are anchored
 * to MANT_MULT_F = 29. Lowering MANT_MULT_ROUND_FRAC therefore only *zeroes*
 * low bits; it never narrows a declared type. The only place the RTL exploits
 * those zeros is bf16_linear_approx's FRAC_ZERO_LSBS, which trims the a*x
 * multiplier and then immediately re-widens the product before the subtract.
 *
 * This header breaks that coupling: the log2(e) product, the unified shift
 * register, the polynomial input x and the b - a*x accumulator each get their
 * own independent width knob, so the real minimum of each can be measured.
 *
 * Nothing here is used by production code. It is an analysis model only, kept
 * separate so that src/approximations/bf16_exp2*.hpp stay untouched.
 *
 * Naming maps onto the RTL as follows:
 *   MANT_MULT_F  -> bf16_log2e_mult   MANT_MULT_ROUND_FRAC
 *   UNIFIED_F    -> bf16_unified_shift IN_F
 *   X_F          -> bf16_linear_approx IN_F - FRAC_ZERO_LSBS  (MULT_X_W)
 *   A_F / B_F    -> bf16_coeff_rom     COEFF_F, split per coefficient
 *   CALC_F       -> bf16_linear_approx CALC_F
 *   STATIC_NORM  -> bf16_normalize replaced by a constant shift
 */

#include "../utils/fp_utils.hpp"
#include "bf16_exp2_core.hpp"
#include "ac_int.h"
#include "ac_fixed.h"

namespace pwl_optim {

/** @brief Exact wide accumulator for the b - a*x datapath. */
typedef __int128 acc_t;

// ---------------------------------------------------------------------------
// Coefficient handling
// ---------------------------------------------------------------------------

/**
 * @brief Re-quantise a coefficient from @p from_f to @p to_f fractional bits,
 *        rounding to nearest (ties up).
 *
 * The LUT is a constant table, so the rounding mode costs nothing in hardware:
 * whatever value ends up in the ROM was chosen offline.
 */
inline uint64_t requantise(uint64_t raw, int from_f, int to_f) {
    if (to_f >= from_f) return raw << (to_f - from_f);
    const int d = from_f - to_f;
    return (raw + (1ull << (d - 1))) >> d;
}

/**
 * @brief Optional replacement coefficient table.
 *
 * When supplied, @ref core_approx uses these raw integers instead of
 * re-quantising the shipped 1.20 table. Values are interpreted at the
 * configuration's A_F / B_F scale. Used to evaluate refitted coefficients.
 */
struct CoeffTable {
    uint64_t a[bf16_cfg::LUT_SIZE] = {};
    uint64_t b[bf16_cfg::LUT_SIZE] = {};
};

/** @brief Fetch the shipped 1.20 coefficients for one segment. */
inline void shipped_coeffs(int idx, uint64_t& a_raw, uint64_t& b_raw) {
    ac_int<bf16_exp2_packed::PACKED_W, false> packed = bf16_exp2_packed::coeffs[idx];
    a_raw = 0;
    b_raw = 0;
    for (int i = 0; i < bf16_cfg::COEFF_W; ++i) {
        if (packed[i])                      a_raw |= (1ull << i);
        if (packed[i + bf16_cfg::COEFF_W])  b_raw |= (1ull << i);
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * @brief Datapath width configuration.
 *
 * @tparam MANT_MULT_F_ Fractional bits kept after the log2(e) multiply.
 *                      Production/full precision = 29.
 * @tparam MANT_MULT_Q_ Quantisation mode for that narrowing.
 *                      AC_RND_CONV = RNE (matches RTL), AC_TRN = truncation,
 *                      AC_RND = round half away from zero.
 * @tparam UNIFIED_F_   Fractional bits of the unified shift register.
 *                      Production = IN_F = 38.
 * @tparam X_F_         Fractional bits of x entering the a*x multiplier.
 *                      Production = IN_F = 38. Must be <= UNIFIED_F_.
 * @tparam CALC_F_      Fractional bits kept in the b - a*x result.
 *                      Production = CALC_F = 58.
 * @tparam STATIC_NORM_ Replace the priority encoder + barrel shifter with the
 *                      constant shift of 1 that the value range implies.
 * @tparam A_F_         Fractional bits of the slope coefficient a.
 *                      Production = COEFF_F = 20.
 * @tparam B_F_         Fractional bits of the offset coefficient b.
 *                      Production = COEFF_F = 20. Independent of A_F_.
 */
template<int       MANT_MULT_F_ = bf16_cfg::MANT_MULT_F,
         ac_q_mode MANT_MULT_Q_ = AC_RND_CONV,
         int       UNIFIED_F_   = bf16_cfg::IN_F,
         int       X_F_         = bf16_cfg::IN_F,
         int       CALC_F_      = bf16_cfg::CALC_F,
         bool      STATIC_NORM_ = false,
         int       A_F_         = bf16_cfg::COEFF_F,
         int       B_F_         = bf16_cfg::COEFF_F>
struct Cfg {
    static constexpr int       MANT_MULT_F = MANT_MULT_F_;
    static constexpr ac_q_mode MANT_MULT_Q = MANT_MULT_Q_;
    static constexpr int       UNIFIED_F   = UNIFIED_F_;
    static constexpr int       X_F         = X_F_;
    static constexpr int       CALC_F      = CALC_F_;
    static constexpr bool      STATIC_NORM = STATIC_NORM_;
    static constexpr int       A_F         = A_F_;
    static constexpr int       B_F         = B_F_;

    /** @brief Common scale that b and a*x are aligned to before subtracting. */
    static constexpr int PROD_F = (B_F > A_F + X_F) ? B_F : (A_F + X_F);
    /** @brief Bits dropped when narrowing that result down to CALC_F. */
    static constexpr int CDROP = PROD_F - CALC_F;

    /** @brief Unified shift register width (integer part is unchanged). */
    static constexpr int UNIFIED_W = bf16_cfg::IN_CONV_INT_W + UNIFIED_F;
    /** @brief Normalised mantissa width feeding the RNE rounder. */
    static constexpr int NORM_W = CALC_F + 1;
    /** @brief Bits discarded by the final round-to-BF16. */
    static constexpr int BASE_SHIFT = CALC_F - bf16_cfg::TARGET_MANT_W;

    static_assert(X_F <= UNIFIED_F, "X_F must not exceed UNIFIED_F");
    static_assert(X_F >= bf16_cfg::LUT_ADDR_W, "X_F must cover the LUT index");
    static_assert(MANT_MULT_F >= 1 && MANT_MULT_F <= bf16_cfg::MANT_MULT_F,
                  "MANT_MULT_F must be in [1, 29]");
    static_assert(CALC_F > bf16_cfg::TARGET_MANT_W, "CALC_F must exceed the BF16 mantissa");
    static_assert(A_F_ >= 1 && B_F_ >= 1, "coefficient widths must be positive");
};

/** @brief Production configuration: must be bit-identical to bf16_exp2_approx<29>. */
using ProductionCfg = Cfg<>;

/** @brief Configuration currently used by the optimised RTL build. */
using CurrentRtlCfg = Cfg<21>;

// ---------------------------------------------------------------------------
// Instrumentation
// ---------------------------------------------------------------------------

/** @brief Optional observations collected while evaluating the datapath. */
struct Stats {
    int  msb_min       = 1 << 30;  ///< lowest  normalisation MSB index seen
    int  msb_max       = -(1 << 30);  ///< highest normalisation MSB index seen
    int  static_norm_violations = 0;  ///< times msb_idx != CALC_F - 1
    int  negative_res  = 0;        ///< times b - a*x went negative
    int  zero_res      = 0;        ///< times b - a*x was exactly zero
    uint64_t max_ax    = 0;        ///< largest raw a*x product seen
    uint64_t max_res   = 0;        ///< largest raw b - a*x seen
    uint64_t min_res   = ~0ull;    ///< smallest raw b - a*x seen
    int  x_min_zero_lsbs = 1 << 30;  ///< guaranteed zero LSBs of the unified fraction
    uint64_t max_a_raw = 0;        ///< largest re-quantised a actually used
    uint64_t min_a_raw = ~0ull;    ///< smallest re-quantised a actually used
    uint64_t max_b_raw = 0;        ///< largest re-quantised b actually used
    uint64_t min_b_raw = ~0ull;    ///< smallest re-quantised b actually used

    void reset() { *this = Stats(); }
};

// ---------------------------------------------------------------------------
// Back end: normalise + round to BF16
// ---------------------------------------------------------------------------

/**
 * @brief Normalisation and RNE rounding, split out so that a coefficient
 *        refit can probe it directly.
 *
 * @param res            b - a*x as an unsigned integer at CALC_F scale.
 * @param exponent_bias  integer part of the shifted unified value, negated.
 * @param st             optional instrumentation sink.
 */
template<class CFG>
inline FPRaw round_to_bf16(acc_t res, int32_t exponent_bias, Stats* st = nullptr) {
    constexpr int CF = CFG::CALC_F;

    // b - a*x stays in [0.5, 1) for the shipped coefficient table, so the MSB
    // sits at CALC_F - 1 on every input. STATIC_NORM asserts that and replaces
    // the priority encoder + barrel shifter with a fixed shift of 1.
    int msb_idx = -1;
    if constexpr (CFG::STATIC_NORM) {
        msb_idx = CF - 1;
    } else {
        for (int i = CF; i >= 0; --i) { if ((res >> i) & 1) { msb_idx = i; break; } }
    }

    if (st) {
        int actual = -1;
        for (int i = CF; i >= 0; --i) { if ((res >> i) & 1) { actual = i; break; } }
        if (actual < st->msb_min) st->msb_min = actual;
        if (actual > st->msb_max) st->msb_max = actual;
        if (actual != CF - 1) st->static_norm_violations++;
    }

    FPRaw result = {};
    result.sign = 0;
    if (msb_idx < 0) {
        result.status.is_zero = true;
        result.exponent = 0;
        return result;
    }

    const int32_t poly_exp = msb_idx - CF;
    const acc_t full_mant = (msb_idx <= CF) ? (res << (CF - msb_idx))
                                            : (res >> (msb_idx - CF));

    const int32_t final_exponent = poly_exp + exponent_bias;
    const bool    is_sub    = (final_exponent < bf16_cfg::TARGET_MIN_EXP);
    const int     shift_val = CFG::BASE_SHIFT
                            + (is_sub ? (bf16_cfg::TARGET_MIN_EXP - final_exponent) : 0);

    constexpr int POLY_W = CFG::NORM_W;
    auto bit_at = [&](int i) -> bool {
        return (i >= 0 && i < POLY_W) ? (bool)((full_mant >> i) & 1) : false;
    };

    const bool lsb_bit   = (shift_val < POLY_W) ? bit_at(shift_val) : false;
    const bool guard_bit = (shift_val > 0 && shift_val <= POLY_W) ? bit_at(shift_val - 1) : false;

    bool sticky_bit = false;
    if (shift_val > 1) {
        if (shift_val > POLY_W) {
            sticky_bit = (full_mant != 0);
        } else {
            const acc_t mask = (((acc_t)1) << (shift_val - 1)) - 1;
            sticky_bit = ((full_mant & mask) != 0);
        }
    }

    const bool round_up = guard_bit && (lsb_bit || sticky_bit);

    constexpr uint32_t EXT_MASK = (1u << bf16_cfg::EXT_MANT_W) - 1;
    uint32_t result_m_ext = 0;
    if (shift_val < POLY_W) {
        result_m_ext = (uint32_t)((full_mant >> shift_val) & EXT_MASK);
    }
    if (round_up) result_m_ext = (result_m_ext + 1) & EXT_MASK;

    int32_t adjusted_exp = is_sub ? bf16_cfg::TARGET_MIN_EXP : final_exponent;
    if ((result_m_ext >> bf16_cfg::CARRY_BIT_IDX) & 1) {
        adjusted_exp++;
        result_m_ext >>= 1;
    }

    if (result_m_ext == 0) {
        result.status.is_zero = true;
        result.exponent = 0;
    } else if (is_sub && !((result_m_ext >> bf16_cfg::HIDDEN_BIT_IDX) & 1)) {
        result.mantissa = result_m_ext & 0x7F;
        result.hidden_bit = 0;
        result.exponent = bf16_cfg::TARGET_MIN_EXP - 1;
        result.status.is_denormal = true;
    } else {
        result.mantissa = result_m_ext & 0x7F;
        result.hidden_bit = 1;
        result.exponent = adjusted_exp;
        result.status.is_denormal = false;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Front end: everything up to the polynomial input
// ---------------------------------------------------------------------------

/** @brief Result of the mantissa preparation, base conversion and shift. */
struct FrontEnd {
    uint64_t x = 0;              ///< polynomial input, X_F fractional bits
    int32_t  exponent_bias = 0;  ///< negated integer part of the shifted value
    int      lut_index = 0;      ///< segment index, already inverted for 2^-x
};

/** @brief Mantissa prep, optional log2(e) multiply, unified shift, LUT index. */
template<class CFG>
inline FrontEnd front_end(const FPRaw& in, bool base2, Stats* st = nullptr) {
    constexpr int MMF = CFG::MANT_MULT_F;
    constexpr int UF  = CFG::UNIFIED_F;
    constexpr int XF  = CFG::X_F;

    // --- 1. Mantissa source: Q1.7, always in [1.0, 2.0) -------------------
    ac_fixed<bf16_cfg::MANT_SRC_W, bf16_cfg::MANT_SRC_I, false> mant_src;
    mant_src[bf16_cfg::MANT_SRC_W - 1] = 1;
    mant_src.set_slc(0, (ac_int<bf16_cfg::TARGET_MANT_W, false>)in.mantissa);

    // --- 2. Base conversion: mant_src * log2(e), then quantise to 2.MMF ---
    ac_fixed<bf16_cfg::LOG2E_W, bf16_cfg::LOG2E_I, false> log2e_const;
    log2e_const.set_slc(0, bf16_exp2_packed::log2e_int_val);

    ac_fixed<bf16_cfg::MANT_MULT_W, bf16_cfg::MANT_MULT_I, false>
        mant_mult = mant_src * log2e_const;

    ac_fixed<bf16_cfg::MANT_MULT_I + MMF, bf16_cfg::MANT_MULT_I, false, CFG::MANT_MULT_Q>
        mant_mult_q = mant_mult;

    // --- 3. Unified format + exponent shift -------------------------------
    // Narrowing UNIFIED_F below IN_F is exact as long as UNIFIED_F >= MMF:
    // a left shift moves bits up, and a right shift by at most -INPUT_MIN_EXP
    // only discards bits that the X_F slice would drop anyway.
    typedef ac_fixed<CFG::UNIFIED_W, bf16_cfg::IN_CONV_INT_W, false> unified_t;
    unified_t val = base2 ? (unified_t)mant_src : (unified_t)mant_mult_q;

    const int32_t e = in.exponent;
    if (e >= 0) val <<= e;
    else        val >>= (-e);

    FrontEnd fe;
    fe.exponent_bias = -(int)val.to_int();

    // --- 4. Polynomial input x: top X_F bits of the fraction (truncation) --
    ac_int<XF, false> x_slice = val.template slc<XF>(UF - XF);
    for (int i = 0; i < XF; ++i) { if (x_slice[i]) fe.x |= (1ull << i); }

    if (st) {
        ac_int<UF, false> frac_full = val.template slc<UF>(0);
        int z = UF;
        for (int i = 0; i < UF; ++i) { if (frac_full[i]) { z = i; break; } }
        if (z < st->x_min_zero_lsbs) st->x_min_zero_lsbs = z;
    }

    // --- 5. Segment index from the top LUT_ADDR_W bits of x ---------------
    const int raw_index = (int)(fe.x >> (XF - bf16_cfg::LUT_ADDR_W));
    fe.lut_index = bf16_cfg::LUT_MAX_IDX - raw_index;

    return fe;
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

/**
 * @brief Width-parameterised equivalent of bf16_exp2_core_approx.
 *
 * Structurally identical to production: mantissa prep, optional log2(e)
 * multiply, unified shift, LUT lookup, b - a*x, normalise, RNE to BF16.
 * Only the intermediate widths differ.
 *
 * @param in     Decomposed BF16 input (already known to be in the core range).
 * @param base2  true -> 2^x, false -> e^x.
 * @param st     Optional instrumentation sink.
 * @param tbl    Optional refitted coefficient table. When null the shipped
 *               1.20 table is re-quantised to A_F / B_F.
 */
template<class CFG>
inline FPRaw core_approx(const FPRaw& in, bool base2,
                         Stats* st = nullptr, const CoeffTable* tbl = nullptr) {
    constexpr int XF = CFG::X_F;
    constexpr int AF = CFG::A_F;
    constexpr int BF = CFG::B_F;

    const FrontEnd fe = front_end<CFG>(in, base2, st);

    // --- 6. Coefficients, narrowed to their own independent widths --------
    uint64_t a, b;
    if (tbl) {
        a = tbl->a[fe.lut_index];
        b = tbl->b[fe.lut_index];
    } else {
        uint64_t a20, b20;
        shipped_coeffs(fe.lut_index, a20, b20);
        a = requantise(a20, bf16_cfg::COEFF_F, AF);
        b = requantise(b20, bf16_cfg::COEFF_F, BF);
    }

    if (st) {
        if (a > st->max_a_raw) st->max_a_raw = a;
        if (a < st->min_a_raw) st->min_a_raw = a;
        if (b > st->max_b_raw) st->max_b_raw = b;
        if (b < st->min_b_raw) st->min_b_raw = b;
    }

    // --- 7. b - a*x, both operands aligned to the common PROD_F scale -----
    // a*x carries A_F + X_F fractional bits, b carries B_F. Whichever is
    // wider sets the scale; the other is shifted up. No bits are lost here.
    const acc_t ax    = ((acc_t)a * (acc_t)fe.x) << (CFG::PROD_F - AF - XF);
    const acc_t b_ext = ((acc_t)b) << (CFG::PROD_F - BF);
    acc_t res_signed  = b_ext - ax;

    if (st) {
        if ((uint64_t)(ax >> (CFG::PROD_F - AF - XF)) > st->max_ax)
            st->max_ax = (uint64_t)(ax >> (CFG::PROD_F - AF - XF));
        if (res_signed < 0) st->negative_res++;
        else {
            const uint64_t r = (uint64_t)res_signed;
            if (r == 0) st->zero_res++;
            if (r > st->max_res) st->max_res = r;
            if (r < st->min_res) st->min_res = r;
        }
    }
    if (res_signed < 0) res_signed = 0;

    // --- 8. Narrow the accumulator to CALC_F fractional bits --------------
    const acc_t res = (CFG::CDROP >= 0) ? (res_signed >> CFG::CDROP)
                                        : (res_signed << (-CFG::CDROP));

    // --- 9. Normalise and round to BF16 -----------------------------------
    return round_to_bf16<CFG>(res, fe.exponent_bias, st);
}

/**
 * @brief Width-parameterised equivalent of bf16_exp2_approx.
 *
 * The special-case ladder is copied verbatim from bf16_exp2.hpp so that the
 * only difference between this and production is the datapath width.
 */
template<class CFG>
inline uint16_t approx(uint16_t raw_input, bool base2 = true,
                       Stats* st = nullptr, const CoeffTable* tbl = nullptr) {
    FPRaw input_parts = fp_decompose(static_cast<uint32_t>(raw_input), FPType::BF16);

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
    } else if (input_parts.exponent < bf16_cfg::INPUT_MIN_EXP) {
        set_plus_one = true;
    } else if (input_parts.exponent > bf16_cfg::INPUT_MAX_EXP) {
        set_plus_zero = true;
    } else {
        core_result = core_approx<CFG>(input_parts, base2, st, tbl);
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

} // namespace pwl_optim

#endif // BF16_EXP2_PWL_OPTIM_HPP

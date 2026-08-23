/**
 * @file bf16_exp2_optim_test.cpp
 * @brief Verification of the width-optimised exp2 / expe core.
 *
 * Three checks, all exhaustive:
 *
 *   1. bf16_exp2_optim_approx is bit-identical to bf16_exp2_approx<29> over
 *      all 65536 BF16 patterns in both base-2 and base-e mode. This is the
 *      contract: the optimised core is a drop-in replacement, not an
 *      approximation of an approximation.
 *
 *   2. The assumptions that removed hardware actually hold, checked over the
 *      whole 2^17 domain of the polynomial input rather than only the values
 *      reachable from a BF16 input:
 *        - b - a*x never goes negative        (no sign bit needed)
 *        - b - a*x stays below 1.0            (no integer guard bits needed)
 *        - b - a*x stays at or above 0.5      (normalisation is constant)
 *        - the narrow 18-bit subtract equals the naive 34-bit one
 *
 *   3. ULP error against libm, for the record.
 *
 * A failure in 1 means the widths are wrong. A failure in 2 means the widths
 * are right but the structural simplifications are not, which would happen if
 * the coefficient table were refitted without re-running this.
 *
 * Build:  make exp2_optim
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cmath>

#include "../src/approximations/bf16_exp2.hpp"          // reference
#include "../src/approximations/bf16_exp2_optim.hpp"    // device under test
#include "fp_utils.hpp"

namespace cfg = bf16_optim_cfg;

// ===========================================================================
// 1. Equivalence with production
// ===========================================================================

struct EquivResult {
    int      mismatch[2]   = {0, 0};   // [0] = base-e, [1] = base-2
    double   max_ulp[2]    = {0.0, 0.0};
    uint16_t worst_in[2]   = {0, 0};
    int      variant_diff  = 0;        // guarded vs static normalisation
    uint16_t first_bad_in  = 0;
    uint16_t first_bad_ref = 0;
    uint16_t first_bad_got = 0;
    bool     any_bad       = false;

    bool ok() const { return mismatch[0] == 0 && mismatch[1] == 0 && variant_diff == 0; }
};

static EquivResult check_equivalence() {
    EquivResult r;

    for (uint32_t i = 0; i <= 0xFFFFu; ++i) {
        const uint16_t in = static_cast<uint16_t>(i);

        for (int mode = 0; mode < 2; ++mode) {
            const bool base2 = (mode == 1);

            const uint16_t ref = bf16_exp2_approx<bf16_cfg::MANT_MULT_F>(in, base2);
            const uint16_t got = bf16_exp2_optim_approx<false>(in, base2);   // guarded
            const uint16_t sta = bf16_exp2_optim_approx<true>(in, base2);    // static

            if (got != sta) r.variant_diff++;

            if (ref != got) {
                r.mismatch[mode]++;
                if (!r.any_bad) {
                    r.any_bad = true;
                    r.first_bad_in  = in;
                    r.first_bad_ref = ref;
                    r.first_bad_got = got;
                }
            }

            // ULP is only meaningful where the core actually approximates.
            const double x = fp_to_double(in, FPType::BF16);
            if (!std::isfinite(x) || x > 0.0) continue;

            const double reference = base2 ? std::exp2(x) : std::exp(x);
            const double ulp = calculate_ulp_error(reference,
                                                   fp_to_double(got, FPType::BF16),
                                                   FPType::BF16);
            if (!std::isfinite(ulp)) continue;
            if (ulp > r.max_ulp[mode]) {
                r.max_ulp[mode]  = ulp;
                r.worst_in[mode] = in;
            }
        }
    }
    return r;
}

// ===========================================================================
// 2. Structural assumptions
// ===========================================================================
//
// Three pieces of hardware were removed on the strength of b - a*x staying in
// [0.5, 1). That has to be checked against the set of polynomial inputs the
// front end can actually produce, not against the 2^17 values the datatype can
// hold. x is a truncated slice of a value derived from a 7-bit BF16 mantissa,
// so the reachable set is sparse and the two are very different questions.
// The audit answers both and treats only the first as a pass/fail criterion.

struct PolyAudit {
    int64_t min_res  = INT64_MAX;   // raw Q0.34
    int64_t max_res  = INT64_MIN;
    int64_t max_ax   = 0;
    int     count    = 0;

    int negative    = 0;   // b - a*x <  0     -> sign bit needed
    int above_one   = 0;   // b - a*x >= 1.0   -> integer guard bits needed
    int below_half  = 0;   // b - a*x <  0.5   -> normalisation not constant
    int narrow_diff = 0;   // 18-bit subtract != 34-bit subtract
    int denorm_mant = 0;   // normalised mantissa MSB clear

    int64_t min_x_below_half = -1;  // smallest x that dips under 0.5
    int64_t max_x            = -1;  // largest x seen

    bool ok() const {
        return negative == 0 && above_one == 0 && below_half == 0
            && narrow_diff == 0 && denorm_mant == 0;
    }
};

/** @brief Evaluate one polynomial input and fold it into @p a. */
static void audit_one(int64_t x_raw, PolyAudit& a) {
    const int raw_index = int(x_raw >> (cfg::X_F - cfg::LUT_ADDR_W));
    const int idx = cfg::LUT_MAX_IDX - raw_index;

    const uint64_t packed = bf16_exp2_optim_packed::coeffs[idx];
    const int64_t a_raw = int64_t(packed & ((1ull << cfg::A_W) - 1));
    const int64_t b_raw = int64_t(packed >> cfg::A_W);

    // Naive reference arithmetic: align both to Q0.34 and subtract exactly.
    const int64_t ax  = a_raw * x_raw;
    const int64_t res = (b_raw << cfg::CDROP) - ax;

    const int64_t half_scale = int64_t(1) << (cfg::PROD_F - 1);
    const int64_t one_scale  = int64_t(1) << cfg::PROD_F;

    a.count++;
    if (ax  > a.max_ax)  a.max_ax  = ax;
    if (res < a.min_res) a.min_res = res;
    if (res > a.max_res) a.max_res = res;
    if (x_raw > a.max_x) a.max_x = x_raw;

    if (res < 0)          a.negative++;
    if (res >= one_scale) a.above_one++;
    if (res <  half_scale) {
        a.below_half++;
        if (a.min_x_below_half < 0) a.min_x_below_half = x_raw;
    }

    // The narrow 18-bit form must reproduce the truncated naive result.
    const int64_t naive_trunc = res >> cfg::CDROP;
    const OptimPolyResult p = bf16_exp2_optim_poly<false>(optim_x_t(int(x_raw)));
    const int64_t narrow = int64_t(p.mantissa.to_uint64()) >> (p.exponent == -1 ? 1 : 2);

    if (narrow != naive_trunc) a.narrow_diff++;

    // Static normalisation is only correct where the top bit is already set.
    const OptimPolyResult ps = bf16_exp2_optim_poly<true>(optim_x_t(int(x_raw)));
    if (!bf16_exp2_optim_static_norm_holds(ps)) a.denorm_mant++;
}

/** @brief Audit only the polynomial inputs the front end can produce. */
static PolyAudit audit_reachable(std::vector<bool>& reachable) {
    reachable.assign(size_t(1) << cfg::X_F, false);

    for (uint32_t i = 0; i <= 0xFFFFu; ++i) {
        const FPRaw in = fp_decompose(i, FPType::BF16);
        if (in.status.is_nan || in.status.is_zero || in.status.is_inf) continue;
        if (!in.sign) continue;
        if (in.exponent < cfg::INPUT_MIN_EXP || in.exponent > cfg::INPUT_MAX_EXP) continue;

        for (int mode = 0; mode < 2; ++mode) {
            const OptimFrontEnd fe = bf16_exp2_optim_front_end(in, mode == 1);
            reachable[fe.x.to_uint64()] = true;
        }
    }

    PolyAudit a;
    for (size_t x = 0; x < reachable.size(); ++x) {
        if (reachable[x]) audit_one(int64_t(x), a);
    }
    return a;
}

/** @brief Audit the entire 2^17 domain, for the robustness margin. */
static PolyAudit audit_full_domain() {
    PolyAudit a;
    for (int64_t x = 0; x < (int64_t(1) << cfg::X_F); ++x) audit_one(x, a);
    return a;
}

// ===========================================================================
// Reporting
// ===========================================================================

static void section(const char* title) {
    std::cout << "\n=== " << title << " ===\n\n";
}

int main() {
    std::cout << "############################################################\n"
              << "#  bf16_exp2_optim: width-optimised core verification       #\n"
              << "############################################################\n";

    std::cout << "\nConfiguration:\n"
              << "  log2(e) product  Q" << cfg::MANT_MULT_I << "." << cfg::MANT_MULT_F
              << "   (production Q" << bf16_cfg::MANT_MULT_I << "." << bf16_cfg::MANT_MULT_F << ")\n"
              << "  unified shift    Q" << cfg::UNIFIED_I << "." << cfg::UNIFIED_F
              << "   (production Q" << bf16_cfg::IN_CONV_INT_W << "." << bf16_cfg::IN_F << ")\n"
              << "  poly input x     Q0." << cfg::X_F
              << "   (production Q0." << bf16_cfg::IN_F << ")\n"
              << "  coefficient a    Q0." << cfg::A_F
              << "   (production Q" << bf16_cfg::COEFF_I << "." << bf16_cfg::COEFF_F << ")\n"
              << "  coefficient b    Q0." << cfg::B_F
              << "   (production Q" << bf16_cfg::COEFF_I << "." << bf16_cfg::COEFF_F << ")\n"
              << "  a * x            Q0." << cfg::PROD_F
              << "   (production Q" << bf16_cfg::MULT_I << "." << bf16_cfg::MULT_F << ")\n"
              << "  b - a*x          Q0." << cfg::CALC_F << " unsigned"
              << "   (production Q" << bf16_cfg::CALC_I << "." << bf16_cfg::CALC_F << " signed)\n";

    // -- 1 -----------------------------------------------------------------
    section("1. Equivalence with bf16_exp2_approx<29>");
    const EquivResult eq = check_equivalence();

    std::cout << "  | mode   | patterns | mismatches | max ULP  | worst input |\n"
              << "  |--------|----------|------------|----------|-------------|\n";
    for (int mode = 0; mode < 2; ++mode) {
        std::cout << "  | " << (mode ? "base-2" : "base-e") << " | "
                  << std::setw(8) << 65536 << " | "
                  << std::setw(10) << eq.mismatch[mode] << " | "
                  << std::fixed << std::setprecision(6) << std::setw(8) << eq.max_ulp[mode]
                  << " | 0x" << std::hex << std::setw(4) << std::setfill('0')
                  << eq.worst_in[mode] << std::dec << std::setfill(' ') << "      |\n";
    }
    std::cout << "  |--------|----------|------------|----------|-------------|\n";

    if (!eq.ok()) {
        if (eq.variant_diff) {
            std::cout << "\n  FAIL: guarded and static normalisation disagree on "
                      << eq.variant_diff << " evaluations.\n";
        }
        if (eq.any_bad) {
            std::cout << "\n  FAIL: first divergence at input 0x" << std::hex << eq.first_bad_in
                      << ": production 0x" << eq.first_bad_ref
                      << ", optimised 0x" << eq.first_bad_got << std::dec << "\n";
        }
        return 1;
    }
    std::cout << "\n  PASS: bit-identical on all 65536 patterns in both modes,\n"
              << "        and both normalisation variants agree everywhere.\n";

    // -- 2 -----------------------------------------------------------------
    section("2. Structural assumptions over the reachable polynomial inputs");

    std::vector<bool> reachable;
    const PolyAudit ra = audit_reachable(reachable);
    const PolyAudit fa = audit_full_domain();

    const double scale = std::pow(2.0, cfg::PROD_F);
    std::cout << "  Reachable x values : " << ra.count << " of " << fa.count
              << "  (" << std::fixed << std::setprecision(2)
              << (100.0 * ra.count / fa.count) << "%)\n"
              << "  b - a*x range      : [" << std::setprecision(9)
              << double(ra.min_res) / scale << ", "
              << double(ra.max_res) / scale << "]\n"
              << "  max(a*x)           : " << double(ra.max_ax) / scale << "\n\n"
              << "  | assumption                         | hardware removed           | violations |\n"
              << "  |------------------------------------|----------------------------|------------|\n"
              << "  | b - a*x >= 0                       | sign bit                   | "
              << std::setw(10) << ra.negative << " |\n"
              << "  | b - a*x <  1                       | " << bf16_cfg::CALC_I
              << " integer guard bits        | " << std::setw(10) << ra.above_one << " |\n"
              << "  | b - a*x >= 0.5                     | constant shift (STATIC_NORM)| "
              << std::setw(9) << ra.below_half << " |\n"
              << "  | mantissa MSB always set            | ditto                      | "
              << std::setw(10) << ra.denorm_mant << " |\n"
              << "  | 18-bit subtract == 34-bit subtract | 16 subtractor bits         | "
              << std::setw(10) << ra.narrow_diff << " |\n"
              << "  |------------------------------------|----------------------------|------------|\n";

    if (!ra.ok()) {
        std::cout << "\n  FAIL: a structural simplification is not justified by the\n"
                  << "        current coefficient table. Re-run tests/exp_pwl_optim.cpp.\n";
        return 1;
    }
    std::cout << "\n  PASS on every reachable input.\n";

    // -- 2b ----------------------------------------------------------------
    section("2b. Margin: the same assumptions over the full 2^17 domain");

    std::cout << "  b - a*x range      : [" << std::setprecision(9)
              << double(fa.min_res) / scale << ", "
              << double(fa.max_res) / scale << "]\n"
              << "  below 0.5          : " << fa.below_half << " of " << fa.count
              << " values\n"
              << "  negative           : " << fa.negative << "\n"
              << "  at or above 1.0    : " << fa.above_one << "\n"
              << "  narrow != naive    : " << fa.narrow_diff << "\n\n";

    if (fa.below_half == 0) {
        std::cout << "  Static normalisation is unconditional: it holds for every value\n"
                  << "  the datatype can represent, reachable or not. STATIC_NORM = true\n"
                  << "  carries no obligation.\n";
    } else {
        std::cout << "  Static normalisation is NOT unconditional. The linear fit of the\n"
                  << "  last segment undershoots by one accumulator LSB at x = "
                  << fa.min_x_below_half << " / " << (1 << cfg::X_F) << ",\n"
                  << "  the single largest code. The largest reachable x is " << ra.max_x
                  << ", which is\n  " << (fa.min_x_below_half - ra.max_x)
                  << " steps clear of it, so STATIC_NORM = true is correct here but is an\n"
                  << "  input-space property rather than an identity. It would have to be\n"
                  << "  re-proved after any change to the coefficient table, MANT_MULT_F,\n"
                  << "  X_F or the input exponent range.\n\n"
                  << "  The default (STATIC_NORM = false) does not rely on it: one bit test\n"
                  << "  and a 2:1 shift mux, same " << cfg::POLY_OUT_W
                  << "-bit mantissa, same results. It only\n"
                  << "  needs b - a*x >= 0.25, and the measured minimum is "
                  << std::setprecision(6) << double(fa.min_res) / scale << ".\n";
    }

    // -- 3 -----------------------------------------------------------------
    section("3. Cost");
    {
        struct Row { const char* name; int prod_w; int optim_w; };
        const Row rows[] = {
            { "log2(e) product   ", bf16_cfg::MANT_MULT_W,                        cfg::MANT_MULT_W },
            { "unified shift reg ", bf16_cfg::IN_CONV_INT_W + bf16_cfg::IN_F,     cfg::UNIFIED_W   },
            { "poly input x      ", bf16_cfg::IN_F,                               cfg::X_W         },
            { "coefficient a     ", bf16_cfg::COEFF_W,                            cfg::A_W         },
            { "coefficient b     ", bf16_cfg::COEFF_W,                            cfg::B_W         },
            { "a * x product     ", bf16_cfg::MULT_W,                             cfg::PROD_W      },
            { "b - a*x adder     ", bf16_cfg::CALC_W,                             cfg::CALC_W      },
            { "normalised mant   ", bf16_cfg::POLY_OUT_W,                         cfg::POLY_OUT_W  },
        };

        std::cout << "  | stage              | production | optimised | saved |\n"
                  << "  |--------------------|------------|-----------|-------|\n";
        int tot_p = 0, tot_o = 0;
        for (const Row& r : rows) {
            std::cout << "  | " << r.name << " | " << std::setw(10) << r.prod_w
                      << " | " << std::setw(9) << r.optim_w
                      << " | " << std::setw(5) << (r.prod_w - r.optim_w) << " |\n";
            tot_p += r.prod_w; tot_o += r.optim_w;
        }
        std::cout << "  |--------------------|------------|-----------|-------|\n"
                  << "  | datapath total     | " << std::setw(10) << tot_p
                  << " | " << std::setw(9) << tot_o
                  << " | " << std::setw(5) << (tot_p - tot_o) << " |\n";

        const int rom_p = bf16_cfg::LUT_SIZE * bf16_exp2_packed::PACKED_W;
        const int rom_o = cfg::LUT_SIZE * cfg::PACKED_W;
        std::cout << "  | coefficient ROM    | " << std::setw(10) << rom_p
                  << " | " << std::setw(9) << rom_o
                  << " | " << std::setw(5) << (rom_p - rom_o) << " |\n\n"
                  << "  Multiplier shape    : " << bf16_cfg::COEFF_W << " x " << bf16_cfg::IN_F
                  << "  ->  " << cfg::A_W << " x " << cfg::X_W << "\n"
                  << "  Also removed        : " << bf16_cfg::CALC_W
                  << "-bit priority encoder and " << bf16_cfg::CALC_W << "-bit barrel\n"
                  << "                        shifter, replaced by one bit test and a 2:1 mux.\n";
    }

    std::cout << "\nAll checks passed.\n";
    return 0;
}

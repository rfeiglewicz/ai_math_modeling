/**
 * @file exp_pwl_optim.cpp
 * @brief Datapath width optimisation study for bf16_exp2_approx.
 *
 * Answers, exhaustively over all 65536 BF16 patterns in both base-2 and
 * base-e mode:
 *
 *   1. Is the width-parameterised replica bit-identical to production at
 *      production settings?  (sanity gate for everything below)
 *   2. How far can the log2(e) product be narrowed, per quantisation mode?
 *   3. Does narrowing it actually shrink anything downstream?
 *   4. How narrow can the unified shift register get?
 *   5. How narrow can the polynomial input x get?
 *   6. How narrow can the b - a*x accumulator get?
 *   7. Where is the joint (x, calc) Pareto frontier?
 *   8. Is the normalisation shift a constant?
 *   9. What is the resulting minimal configuration?
 *  10. How narrow can the LUT coefficients a and b get, independently?
 *
 * The pass criterion is bit-exactness against bf16_exp2_approx<29> in both
 * modes, which is strictly stronger than the <= 0.5 ULP budget. Section 10
 * also reports the weaker <= 0.5 ULP verdict, because narrowing a constant
 * ROM changes the approximation itself rather than just its arithmetic.
 *
 * Build:  make exp_pwl_optim
 * Run:    ./build/exp_pwl_optim
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <cmath>

#include "../src/approximations/bf16_exp2.hpp"
#include "../src/approximations/bf16_exp2_pwl_optim.hpp"
#include "fp_utils.hpp"

using pwl_optim::Cfg;

// ===========================================================================
// Evaluation
// ===========================================================================

struct Result {
    int      mismatch_e = 0;      // base-e patterns differing from production
    int      mismatch_2 = 0;      // base-2 patterns differing from production
    double   max_ulp_e  = 0.0;
    double   max_ulp_2  = 0.0;
    uint16_t worst_in_e = 0;
    pwl_optim::Stats stats;

    bool bit_exact() const { return mismatch_e == 0 && mismatch_2 == 0; }
};

/** @brief Exhaustive comparison of one configuration against production. */
template<class CFG>
Result evaluate(bool collect_stats = false) {
    Result r;
    pwl_optim::Stats* st = collect_stats ? &r.stats : nullptr;

    for (uint32_t i = 0; i <= 0xFFFFu; ++i) {
        const uint16_t in = static_cast<uint16_t>(i);

        for (int mode = 0; mode < 2; ++mode) {
            const bool base2 = (mode == 1);

            const uint16_t ref = bf16_exp2_approx<bf16_cfg::MANT_MULT_F>(in, base2);
            const uint16_t got = pwl_optim::approx<CFG>(in, base2, st);

            if (ref != got) {
                if (base2) r.mismatch_2++; else r.mismatch_e++;
            }

            const double x = fp_to_double(in, FPType::BF16);
            if (!std::isfinite(x) || x > 0.0) continue;   // model clamps x > 0 to 1.0

            const double reference = base2 ? std::exp2(x) : std::exp(x);
            const double ulp = calculate_ulp_error(reference,
                                                   fp_to_double(got, FPType::BF16),
                                                   FPType::BF16);
            if (!std::isfinite(ulp)) continue;

            if (base2) {
                if (ulp > r.max_ulp_2) r.max_ulp_2 = ulp;
            } else {
                if (ulp > r.max_ulp_e) { r.max_ulp_e = ulp; r.worst_in_e = in; }
            }
        }
    }
    return r;
}

// ===========================================================================
// Reporting
// ===========================================================================

static const char* const RULE =
    "|------------------|-----|-----|-----|-----|-----|-----|------|------|----------|----------|-----------|\n";

static void table_head(const char* first_col) {
    std::cout << "| " << std::left << std::setw(16) << first_col << std::right
              << " | MMF |  UF |  XF |  CF |  AF |  BF | e:mm | 2:mm | e:maxULP | 2:maxULP | verdict   |\n"
              << RULE;
}

static void table_foot() { std::cout << RULE; }

template<class CFG>
Result report(const char* label, bool collect_stats = false) {
    Result r = evaluate<CFG>(collect_stats);
    const char* verdict = r.bit_exact() ? "BIT-EXACT"
                        : ((r.max_ulp_e <= 0.5 && r.max_ulp_2 <= 0.5) ? "<=0.5 ULP" : "FAIL     ");

    std::cout << "| " << std::left << std::setw(16) << label << std::right
              << " | " << std::setw(3) << CFG::MANT_MULT_F
              << " | " << std::setw(3) << CFG::UNIFIED_F
              << " | " << std::setw(3) << CFG::X_F
              << " | " << std::setw(3) << CFG::CALC_F
              << " | " << std::setw(3) << CFG::A_F
              << " | " << std::setw(3) << CFG::B_F
              << " | " << std::setw(4) << r.mismatch_e
              << " | " << std::setw(4) << r.mismatch_2
              << " | " << std::fixed << std::setprecision(6) << std::setw(8) << r.max_ulp_e
              << " | " << std::setw(8) << r.max_ulp_2
              << " | " << verdict << " |\n";
    return r;
}

/** @brief Number of bits needed to hold @p v (0 -> 0). */
static int bits_needed(uint64_t v) {
    int n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}

static void section(const char* title) {
    std::cout << "\n=== " << title << " ===\n\n";
}

// ===========================================================================
// Sweeps
// ===========================================================================

// --- 2. log2(e) product width, per quantisation mode -----------------------
template<int MMF, ac_q_mode Q>
void sweep_mant_mult(const char* tag) {
    report<Cfg<MMF, Q>>(tag);
    if constexpr (MMF > 17) sweep_mant_mult<MMF - 1, Q>(tag);
}

// --- 4. unified shift register width --------------------------------------
template<int UF>
void sweep_unified() {
    report<Cfg<21, AC_RND_CONV, UF, UF>>("unified");
    if constexpr (UF > 17) sweep_unified<UF - 1>();
}

// --- 5. polynomial input width (accumulator left wide) ---------------------
template<int XF>
void sweep_x() {
    report<Cfg<21, AC_RND_CONV, 38, XF, 58>>("x width");
    if constexpr (XF > 15) sweep_x<XF - 1>();
}

// --- 6. accumulator width (x left wide) -----------------------------------
template<int CF>
void sweep_calc() {
    report<Cfg<21, AC_RND_CONV, 38, 30, CF>>("calc width");
    if constexpr (CF > 20) sweep_calc<CF - 1>();
}

// --- 7. joint Pareto frontier ---------------------------------------------
template<int XF, int CF>
void pareto_cell() {
    std::cout << (evaluate<Cfg<21, AC_RND_CONV, 38, XF, CF>>().bit_exact() ? "  ok" : "  . ");
    if constexpr (CF > 15) pareto_cell<XF, CF - 1>();
}

template<int XF>
void pareto_row() {
    std::cout << "  XF = " << std::setw(2) << XF << " |";
    pareto_cell<XF, 24>();
    std::cout << "\n";
    if constexpr (XF > 15) pareto_row<XF - 1>();
}

// --- 10. coefficient widths -----------------------------------------------
// The LUT is a constant ROM, so re-quantising it is free: whatever value ends
// up in the table was chosen offline. What shrinks is the a*x multiplier, the
// b alignment and the ROM itself. a and b are swept independently because
// they enter the datapath at different points: b is the result to first
// order, a only scales x in [0, 1).

template<int AF>
void sweep_a() {
    report<Cfg<21, AC_RND_CONV, 38, 38, 58, false, AF, bf16_cfg::COEFF_F>>("a width");
    if constexpr (AF > 8) sweep_a<AF - 1>();
}

template<int BF>
void sweep_b() {
    report<Cfg<21, AC_RND_CONV, 38, 38, 58, false, bf16_cfg::COEFF_F, BF>>("b width");
    if constexpr (BF > 12) sweep_b<BF - 1>();
}

/** @brief Three-level verdict used by the coefficient grid. */
template<class CFG>
const char* grade() {
    const Result r = evaluate<CFG>();
    if (r.bit_exact()) return "  ok";
    if (r.max_ulp_e <= 0.5 && r.max_ulp_2 <= 0.5) return "   ~";
    return "   .";
}

// The grid is run on the MINIMAL datapath, not the production one: that is the
// design point, and the two do not agree. A coarser accumulator discards part
// of the coefficient quantisation error before it can reach the rounder, so
// some (a, b) pairs that fail at CALC_F = 58 pass at CALC_F = 18.
template<int AF, int BF>
void coeff_cell() {
    std::cout << grade<Cfg<21, AC_RND_CONV, 21, 17, 18, true, AF, BF>>();
    if constexpr (BF > 10) coeff_cell<AF, BF - 1>();
}

template<int AF>
void coeff_row() {
    std::cout << "  AF = " << std::setw(2) << AF << " |";
    coeff_cell<AF, 20>();
    std::cout << "\n";
    if constexpr (AF > 8) coeff_row<AF - 1>();
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    std::cout << "############################################################\n"
              << "#  bf16_exp2 PWL datapath width study                      #\n"
              << "#  criterion: bit-exact vs bf16_exp2_approx<29> (both modes)#\n"
              << "############################################################\n";

    std::cout << "\nProduction widths from bf16_cfg:\n"
              << "  MANT_SRC  Q" << bf16_cfg::MANT_SRC_I << "." << bf16_cfg::MANT_SRC_F
              << "   LOG2E Q" << bf16_cfg::LOG2E_I << "." << bf16_cfg::LOG2E_F
              << "   MANT_MULT Q" << bf16_cfg::MANT_MULT_I << "." << bf16_cfg::MANT_MULT_F << "\n"
              << "  UNIFIED   Q" << bf16_cfg::IN_CONV_INT_W << "." << bf16_cfg::IN_F
              << "   COEFF Q" << bf16_cfg::COEFF_I << "." << bf16_cfg::COEFF_F
              << "   a*x   Q" << bf16_cfg::MULT_I << "." << bf16_cfg::MULT_F << "\n"
              << "  CALC      Q" << bf16_cfg::CALC_I << "." << bf16_cfg::CALC_F
              << " signed   POLY_OUT Q" << bf16_cfg::POLY_OUT_I << "." << bf16_cfg::POLY_OUT_F
              << "   LUT " << bf16_cfg::LUT_SIZE << " x " << bf16_exp2_packed::PACKED_W << " b\n";

    // -- 1 -----------------------------------------------------------------
    section("1. Sanity: parameterised replica vs production");
    table_head("config");
    const Result prod = report<pwl_optim::ProductionCfg>("production", true);
    report<pwl_optim::CurrentRtlCfg>("current RTL");
    table_foot();

    if (!prod.bit_exact()) {
        std::cout << "\nFATAL: replica is not bit-exact at production widths "
                  << "(e:" << prod.mismatch_e << " 2:" << prod.mismatch_2 << ").\n"
                  << "Every result below would be meaningless. Aborting.\n";
        return 1;
    }
    std::cout << "\nReplica matches production on all 65536 patterns in both modes.\n";

    // -- 2 -----------------------------------------------------------------
    section("2. log2(e) product width, per quantisation mode");
    std::cout << "Only the base-e path uses this multiply; base-2 bypasses it entirely.\n\n";
    table_head("mode");
    sweep_mant_mult<29, AC_RND_CONV>("RNE  (conv)");
    table_foot();
    sweep_mant_mult<29, AC_TRN>("TRN  (trunc)");
    table_foot();
    sweep_mant_mult<29, AC_RND>("RND  (half-up)");
    table_foot();

    // -- 3 -----------------------------------------------------------------
    section("3. Does narrowing the product shrink anything downstream?");
    std::cout << "The RTL exploits the zero LSBs via FRAC_ZERO_LSBS = 29 - MMF.\n"
              << "Measured guaranteed zero LSBs of the unified fraction:\n\n"
              << "  |  MMF | 29 - MMF | measured | tight? |\n"
              << "  |------|----------|----------|--------|\n";
    {
        auto probe = [](int mmf, int measured) {
            const int expect = bf16_cfg::MANT_MULT_F - mmf;
            std::cout << "  | " << std::setw(4) << mmf
                      << " | " << std::setw(8) << expect
                      << " | " << std::setw(8) << measured
                      << " | " << (measured == expect ? "yes" : (measured > expect ? "slack" : "UNSAFE"))
                      << "    |\n";
        };
        probe(29, evaluate<Cfg<29>>(true).stats.x_min_zero_lsbs);
        probe(25, evaluate<Cfg<25>>(true).stats.x_min_zero_lsbs);
        probe(21, evaluate<Cfg<21>>(true).stats.x_min_zero_lsbs);
        probe(17, evaluate<Cfg<17>>(true).stats.x_min_zero_lsbs);
    }
    std::cout << "\nNote: those zeros only narrow the a*x multiplier. bf16_linear_approx\n"
              << "re-widens the product before the subtract, so the 62-bit adder,\n"
              << "normaliser and rounder keep their declared width regardless of MMF.\n";

    // -- 4 -----------------------------------------------------------------
    section("4. Unified shift register width");
    std::cout << "Narrowing below IN_F=38 is exact while UNIFIED_F >= MANT_MULT_F.\n\n";
    table_head("stage");
    sweep_unified<30>();
    table_foot();

    // -- 5 -----------------------------------------------------------------
    section("5. Polynomial input x width (accumulator left at 58)");
    table_head("stage");
    sweep_x<24>();
    table_foot();

    // -- 6 -----------------------------------------------------------------
    section("6. Accumulator width (x left at 30)");
    table_head("stage");
    sweep_calc<30>();
    table_foot();

    // -- 7 -----------------------------------------------------------------
    section("7. Joint (x, calc) Pareto frontier, MMF = 21");
    std::cout << "         CF:";
    for (int c = 24; c >= 15; --c) std::cout << std::setw(4) << c;
    std::cout << "\n";
    pareto_row<20>();

    // -- 8 -----------------------------------------------------------------
    section("8. Is the normalisation shift a constant?");
    {
        const pwl_optim::Stats& s = prod.stats;
        const double scale = std::pow(2.0, bf16_cfg::CALC_F);
        std::cout << "  msb_idx range over all inputs, both modes : ["
                  << s.msb_min << ", " << s.msb_max << "]   (CALC_F = "
                  << bf16_cfg::CALC_F << ")\n"
                  << "  b - a*x range                             : ["
                  << std::fixed << std::setprecision(9)
                  << (double)s.min_res / scale << ", "
                  << (double)s.max_res / scale << "]\n"
                  << "  max(a*x)                                  : "
                  << (double)s.max_ax / std::pow(2.0, bf16_cfg::MULT_F) << "\n"
                  << "  negative results / exact zeros            : "
                  << s.negative_res << " / " << s.zero_res << "\n"
                  << "  msb_idx != CALC_F - 1 occurrences         : "
                  << s.static_norm_violations << "\n\n";

        if (s.msb_min == s.msb_max && s.static_norm_violations == 0) {
            std::cout << "  => b - a*x never leaves [0.5, 1). The priority encoder and the\n"
                      << "     normalising barrel shifter compute a constant: shift = 1.\n"
                      << "  => a*x < 0.5 always, so MULT_I = " << bf16_cfg::MULT_I
                      << " carries " << bf16_cfg::MULT_I << " unused integer bit(s).\n"
                      << "  => b - a*x is never negative, so CALC_I = " << bf16_cfg::CALC_I
                      << " plus the sign bit carry " << (bf16_cfg::CALC_I) << " unused bit(s).\n\n";

            table_head("static norm");
            report<Cfg<21, AC_RND_CONV, 38, 38, 58, true>>("STATIC_NORM=1");
            report<Cfg<21, AC_RND_CONV, 21, 17, 18, true>>("minimal + static");
            table_foot();
        } else {
            std::cout << "  => normalisation is NOT constant; the encoder must stay.\n";
        }
    }

    // -- 9 -----------------------------------------------------------------
    section("9. Candidate minimal configurations");
    table_head("config");
    report<Cfg<29, AC_RND_CONV, 38, 38, 58>>("production");
    report<Cfg<21, AC_RND_CONV, 38, 38, 58>>("current RTL");
    report<Cfg<21, AC_RND_CONV, 21, 20, 22>>("conservative");
    report<Cfg<21, AC_RND_CONV, 21, 18, 19>>("aggressive");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true>>("MINIMAL");
    std::cout << RULE;
    report<Cfg<21, AC_RND_CONV, 21, 17, 17, true>>("one step over");
    report<Cfg<21, AC_RND_CONV, 21, 16, 18, true>>("one step over");
    report<Cfg<20, AC_RND_CONV, 21, 17, 18, true>>("one step over");
    table_foot();

    // -- 10 ----------------------------------------------------------------
    section("10. Coefficient widths a and b");
    {
        const pwl_optim::Stats& s = prod.stats;
        const double as = std::pow(2.0, bf16_cfg::COEFF_F);
        std::cout << "Shipped table: " << bf16_cfg::LUT_SIZE << " entries, Q"
                  << bf16_cfg::COEFF_I << "." << bf16_cfg::COEFF_F << " each, packed "
                  << bf16_exp2_packed::PACKED_W << " b -> "
                  << (bf16_cfg::LUT_SIZE * bf16_exp2_packed::PACKED_W) << " b ROM.\n\n"
                  << "  Observed value ranges over all inputs, both modes:\n"
                  << "    a in [" << std::fixed << std::setprecision(9)
                  << (double)s.min_a_raw / as << ", " << (double)s.max_a_raw / as
                  << "]   needs " << bits_needed(s.max_a_raw) << " b at F="
                  << bf16_cfg::COEFF_F << "\n"
                  << "    b in [" << (double)s.min_b_raw / as << ", "
                  << (double)s.max_b_raw / as
                  << "]   needs " << bits_needed(s.max_b_raw) << " b at F="
                  << bf16_cfg::COEFF_F << "\n\n";

        const bool a_int_dead = bits_needed(s.max_a_raw) <= bf16_cfg::COEFF_F;
        const bool b_int_dead = bits_needed(s.max_b_raw) <= bf16_cfg::COEFF_F;
        if (a_int_dead && b_int_dead) {
            std::cout << "  => COEFF_I = " << bf16_cfg::COEFF_I
                      << " is dead for both coefficients: every a and every b is < 1.\n"
                      << "     That is " << (2 * bf16_cfg::COEFF_I) << " b per entry, "
                      << (bf16_cfg::LUT_SIZE * 2 * bf16_cfg::COEFF_I)
                      << " b of ROM, free of any accuracy cost.\n"
                      << "     (b -> 1.0 only if it is later rounded up at a small B_F;\n"
                      << "      the sweeps below flag that by needing B_F + 1 bits.)\n\n";
        }
    }

    std::cout << "10a. Slope a narrowed alone, production datapath (b kept at Q1.20):\n\n";
    table_head("stage");
    sweep_a<bf16_cfg::COEFF_F>();
    table_foot();

    std::cout << "\n10b. Offset b narrowed alone, production datapath (a kept at Q1.20):\n\n";
    table_head("stage");
    sweep_b<bf16_cfg::COEFF_F>();
    table_foot();

    std::cout << "\nNote: every FAIL above also breaks the 0.5 ULP budget. Production\n"
              << "already sits at 0.499981 / 0.500000 ULP, so there is no slack between\n"
              << "\"bit-exact\" and \"still correctly rounded\" - the two criteria coincide.\n"
              << "Note also that the sweeps are not monotone: AF 19 and 18 fail while\n"
              << "AF 17 is exact. Re-quantisation error is signed, so a coarser grid can\n"
              << "land closer to the value production happens to round to.\n";

    std::cout << "\n10c. Joint (a, b) frontier on the MINIMAL datapath "
              << "(MMF 21, UF 21, XF 17, CF 18, static norm).\n"
              << "     ok = bit-exact, ~ = <=0.5 ULP only, . = fail\n\n";
    std::cout << "         BF:";
    for (int b = 20; b >= 10; --b) std::cout << std::setw(4) << b;
    std::cout << "\n";
    coeff_row<20>();

    std::cout << "\n10d. Candidate coefficient formats on the minimal datapath:\n\n";
    table_head("config");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 20, 20>>("shipped Q1.20");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 20, 18>>("b 18");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 17, 18>>("a 17 / b 18");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 15, 18>>("a 15 / b 18");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 14, 18>>("a 14 / b 18");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 13, 18>>("a 13 / b 18");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 15, 17>>("a 15 / b 17");
    report<Cfg<21, AC_RND_CONV, 21, 17, 18, true, 15, 16>>("a 15 / b 16");
    table_foot();

    std::cout << "\n  => b needs 18 fractional bits, and only 18: BF 17 fails for every AF.\n"
              << "  => a needs 17 fractional bits to be safe. AF 15 also passes, but AF 16\n"
              << "     between them does not, so 15 is a lucky point of the shipped table\n"
              << "     rather than a margin. Both are exhaustively verified; 17 is the\n"
              << "     value that survives a coefficient refit, 15 is the value that\n"
              << "     survives this exact ROM.\n"
              << "  => neither coefficient ever reaches 1.0, so the integer bit goes away.\n\n"
              << "     ROM per entry : " << bf16_exp2_packed::PACKED_W
              << " b  ->  Q0.17 + Q0.18 = 35 b   (Q0.15 + Q0.18 = 33 b)\n"
              << "     ROM total     : " << (bf16_cfg::LUT_SIZE * bf16_exp2_packed::PACKED_W)
              << " b  ->  " << (bf16_cfg::LUT_SIZE * 35) << " b   ("
              << (bf16_cfg::LUT_SIZE * 33) << " b)\n"
              << "     a * x mult    : " << bf16_cfg::COEFF_W << "x" << bf16_cfg::IN_F
              << " b  ->  17x17 b   (15x17 b)\n";

    // -- summary -----------------------------------------------------------
    section("Summary: minimal bit-exact datapath");
    {
        struct Row { const char* name; int prod_w; int min_w; };
        const Row rows[] = {
            { "log2(e) product   ", bf16_cfg::MANT_MULT_W,                       2 + 21 },
            { "unified shift reg ", bf16_cfg::IN_CONV_INT_W + bf16_cfg::IN_F,    9 + 21 },
            { "poly input x      ", bf16_cfg::IN_F,                              17     },
            { "coeff a           ", bf16_cfg::COEFF_W,                           17     },
            { "coeff b           ", bf16_cfg::COEFF_W,                           18     },
            { "a * x product     ", bf16_cfg::MULT_W,                            17 + 17 },
            { "b - a*x adder     ", bf16_cfg::CALC_W,                            19     },
            { "normalised mant   ", bf16_cfg::POLY_OUT_W,                        19     },
        };
        std::cout << "  Configuration: MMF 21, UF 21, XF 17, CF 18, static norm, a Q0.17, b Q0.18\n"
                  << "  Verified exhaustively: 65536 patterns x 2 modes, bit-exact.\n\n"
                  << "  | stage              | production | minimal | saved |\n"
                  << "  |--------------------|------------|---------|-------|\n";
        int tot_p = 0, tot_m = 0;
        for (const Row& r : rows) {
            std::cout << "  | " << r.name << " | " << std::setw(10) << r.prod_w
                      << " | " << std::setw(7) << r.min_w
                      << " | " << std::setw(5) << (r.prod_w - r.min_w) << " |\n";
            tot_p += r.prod_w; tot_m += r.min_w;
        }
        std::cout << "  |--------------------|------------|---------|-------|\n"
                  << "  | total              | " << std::setw(10) << tot_p
                  << " | " << std::setw(7) << tot_m
                  << " | " << std::setw(5) << (tot_p - tot_m) << " |\n\n";

        const int rom_p = bf16_cfg::LUT_SIZE * bf16_exp2_packed::PACKED_W;
        const int rom_m = bf16_cfg::LUT_SIZE * 35;
        std::cout << "  | coefficient ROM    | " << std::setw(10) << rom_p
                  << " | " << std::setw(7) << rom_m
                  << " | " << std::setw(5) << (rom_p - rom_m) << " |\n\n"
                  << "  Plus: priority encoder and normalising barrel shifter removed entirely.\n"
                  << "  Caveat: these minima hold for the shipped LUT coefficients. The\n"
                  << "  sweeps are non-monotone, so a refit of a and b for the narrow\n"
                  << "  datapath could move them in either direction and must be re-run\n"
                  << "  through this study. pwl_optim::CoeffTable exists for exactly that.\n";
    }

    return 0;
}

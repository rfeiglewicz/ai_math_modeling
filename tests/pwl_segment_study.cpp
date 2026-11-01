/**
 * @file pwl_segment_study.cpp
 * @brief Does the PWL core actually need 128 segments?
 *
 * The 128-entry LUT arrived in commit 4ddd871 ("resize number of entries of
 * lut from 64 to 128 to get max ulp error = 0.5 for both functions"), so the
 * number was measured rather than guessed. But that measurement predates the
 * narrow datapath, and it was made with coefficients fitted by
 * gen_bf16_discrete_minmax.py, which fits over the BF16 values in
 * [-0.996, -0.002] -- not over the 17-bit fixed-point fractions that actually
 * index the LUT. The fit and the hardware disagree about where the segment
 * boundaries are, and that disagreement grows as segments get wider. A
 * verdict of "64 is not enough" taken from that script could easily be an
 * artefact of the fit rather than a property of the approximation.
 *
 * This study settles it without touching the datapath. The coefficient table
 * stays 128 entries wide; to emulate N segments, groups of 128/N consecutive
 * entries are given identical coefficients. That is exactly equivalent to a
 * narrower LUT -- same arithmetic, same index extraction, same rounding --
 * so anything measured here transfers directly to a real N-entry ROM.
 *
 * For each N every segment is refitted from scratch:
 *   1. continuous Chebyshev minimax for 2^-x over the segment, giving a
 *      starting (a, b);
 *   2. exhaustive search of a small integer neighbourhood at the production
 *      quantisation (a in Q0.17, b in Q0.18), scored on the inputs that
 *      really reach that segment.
 *
 * Scoring is the true ULP error against exp2/exp in double, not bit-equality
 * with production -- production *is* the 128-segment table, so bit-equality
 * would be circular.
 *
 * ---------------------------------------------------------------------------
 * Result, and why it does not lead anywhere
 * ---------------------------------------------------------------------------
 *
 * 64 segments hold 0.5 ULP. The measured worst case is 0.499981 (e^x) and
 * 0.500000 (2^x) -- identical to 128, with zero incorrectly rounded results
 * out of 4352 inputs that reach the datapath. 32 segments miss by 3 results
 * per mode (0.5033 ULP) and it degrades from there.
 *
 * That contradicts commit 4ddd871 ("resize number of entries of lut from 64
 * to 128 to get max ulp error = 0.5"). The discrepancy is in the fit, not in
 * the approximation: gen_bf16_discrete_minmax.py fits over BF16 values in
 * [-0.996, -0.002] divided into equal-width bins, while the hardware splits
 * [0, 1) by the top bits of a 17-bit fixed-point fraction. The two partitions
 * are not the same, and the mismatch scales with segment width. At 64
 * segments a fit aligned to the hardware's own boundaries clears 0.5 ULP
 * comfortably.
 *
 * It saves nothing. Full-core synthesis on xc7a200tfbg484-1, base-e only,
 * maximum retiming, against the 128-segment "expe pwl+retime" row:
 *
 *     segments   LUT   FF   SRL   CARRY4   DSP   RAMB18   Fmax     levels
 *     128        343   261   37       14     2        1   177.5 MHz     5
 *      64        342   260   37       14     2        1   177.5 MHz     5
 *
 * One LUT and one flip-flop. The ROM does not shrink because it was never
 * the limit: 128 x 35 b is 4480 bits and a RAMB18 holds 18432, so both
 * depths occupy exactly one block. Block RAM cannot be allocated in
 * fractions. Synthesising the ROM on its own confirms it -- 1 RAMB18 either
 * way. Forcing it into LUTs instead (rom_style = "distributed") costs 64 LUT
 * at 128 entries and 35 LUT at 64, so the segment count only starts to
 * matter if the BRAM is wanted for something else. It is not: the core is
 * limited by its 2 DSP48E1 (370 copies per device), not by memory or LUTs.
 *
 * Follow-up: halving the segments alone saves nothing, but a refitted table
 * also frees the datapath widths. tests/pwl_width_refit_study.cpp does that
 * search at 64 segments, and the core now ships 64 entries with narrower
 * widths (see bf16_exp2_optim_pkg.sv). This study is kept as the record of
 * the segment-count question on its own, at the old widths.
 *
 * Build: make pwl_segment_study
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bf16_exp2_pwl_optim.hpp"
#include "fp_utils.hpp"

using pwl_optim::Cfg;
using pwl_optim::CoeffTable;

namespace {

/** @brief Entries in the shipped table; the emulation subdivides this. */
constexpr int FULL_SEGMENTS = bf16_cfg::LUT_SIZE;   // 128

/** @brief ln(2), for the tangency point of the minimax fit. */
constexpr double LN2 = 0.69314718055994530942;

/**
 * @brief Production width configuration of the optim core.
 *
 * MANT_MULT_F=21, UNIFIED_F=21, X_F=17, CALC_F=18, STATIC_NORM=true,
 * A_F=17, B_F=18 -- the minimum found by tests/exp_pwl_optim.cpp.
 */
typedef Cfg<21, AC_RND_CONV, 21, 17, 18, true, 17, 18> CfgProd;

/**
 * @brief Same, but with the priority encoder restored (STATIC_NORM=false).
 *
 * The static normaliser hard-codes a shift of 1, which is only valid while
 * b - a*x stays in [0.5, 1). Wider segments push the fit away from the curve,
 * so this variant separates "the approximation is too coarse" from "the
 * approximation left the range the normaliser assumes".
 */
typedef Cfg<21, AC_RND_CONV, 21, 17, 18, false, 17, 18> CfgDyn;

/**
 * @brief Production widths with roomier coefficients (a Q0.20, b Q0.21).
 *
 * Separates segment count from coefficient quantisation: if a given N fails
 * here too, the segments are genuinely too wide.
 */
typedef Cfg<21, AC_RND_CONV, 21, 17, 21, true, 20, 21> CfgWide;

/** @brief One input that reaches the polynomial datapath. */
struct Point {
    uint16_t raw;        ///< BF16 input pattern
    bool     base2;      ///< true = 2^x, false = e^x
    double   ref;        ///< exact result, in double
    int      lut_index;  ///< segment it lands in, already inverted
};

/** @brief The function every segment approximates, 2^-x on [0, 1). */
inline double f(double x) { return std::exp2(-x); }

/**
 * @brief Collect every input that actually reaches the core.
 *
 * Mirrors the special-case ladder in pwl_optim::approx: NaN, zero, infinity,
 * positive inputs and out-of-range exponents all short-circuit before the
 * LUT, so they constrain nothing.
 */
template <class CFG>
std::vector<Point> collect_points() {
    std::vector<Point> pts;
    pts.reserve(4096);

    for (uint32_t i = 0; i <= 0xFFFFu; ++i) {
        const uint16_t raw = static_cast<uint16_t>(i);
        const FPRaw p = fp_decompose(static_cast<uint32_t>(raw), FPType::BF16);

        if (p.status.is_nan || p.status.is_zero || p.status.is_inf) continue;
        if (!p.sign) continue;                                  // x > 0 -> +1.0
        if (p.exponent < bf16_cfg::INPUT_MIN_EXP) continue;     // -> +1.0
        if (p.exponent > bf16_cfg::INPUT_MAX_EXP) continue;     // -> +0.0

        const double x = fp_to_double(raw, FPType::BF16);

        for (int m = 0; m < 2; ++m) {
            const bool base2 = (m == 1);
            const pwl_optim::FrontEnd fe = pwl_optim::front_end<CFG>(p, base2);

            Point pt;
            pt.raw       = raw;
            pt.base2     = base2;
            pt.ref       = base2 ? std::exp2(x) : std::exp(x);
            pt.lut_index = fe.lut_index;
            pts.push_back(pt);
        }
    }
    return pts;
}

/** @brief Worst ULP and miss count over a point set, for one (a, b). */
template <class CFG>
double score(const std::vector<const Point*>& grp, CoeffTable& tbl,
             const std::vector<int>& idxs, uint64_t a, uint64_t b,
             int& wrong_out) {
    for (int ix : idxs) { tbl.a[ix] = a; tbl.b[ix] = b; }

    double worst = 0.0;
    int    wrong = 0;
    for (const Point* p : grp) {
        const uint16_t got = pwl_optim::approx<CFG>(p->raw, p->base2, nullptr, &tbl);
        const double   ulp = calculate_ulp_error(p->ref,
                                                 fp_to_double(got, FPType::BF16),
                                                 FPType::BF16);
        if (!std::isfinite(ulp)) continue;
        if (ulp > worst) worst = ulp;
        if (ulp > 0.5)   ++wrong;
    }
    wrong_out = wrong;
    return worst;
}

/** @brief Outcome of refitting the whole table at one segment count. */
struct FitResult {
    double max_ulp_e = 0.0;     ///< worst ULP over e^x
    double max_ulp_2 = 0.0;     ///< worst ULP over 2^x
    int    wrong_e   = 0;       ///< e^x results that are not correctly rounded
    int    wrong_2   = 0;       ///< 2^x results that are not correctly rounded
    int    clipped   = 0;       ///< segments whose fit wanted a >= 1.0 or b >= 1.0
    int    max_radius = 0;      ///< widest search any segment needed
    CoeffTable table;           ///< the fitted coefficients, for ROM emission
};

/**
 * @brief Refit all segments for an emulated N-entry LUT and measure the result.
 *
 * The integer search around the minimax guess grows until the winning
 * candidate is no longer on the boundary, so the answer does not depend on a
 * guessed search width. Without this, a too-small window silently reports a
 * segment count as failing when only the fit was constrained.
 *
 * @param n_seg      emulated segment count; must divide FULL_SEGMENTS
 * @param radius0    initial half-width of the integer search
 * @param radius_max point at which to stop doubling and accept the result
 */
template <class CFG>
FitResult fit_and_measure(int n_seg, const std::vector<Point>& pts,
                          int radius0, int radius_max) {
    FitResult res;
    CoeffTable tbl;

    const int      group = FULL_SEGMENTS / n_seg;
    const uint64_t A_ONE = 1ull << CFG::A_F;   // a must stay below 1.0
    const uint64_t B_ONE = 1ull << CFG::B_F;   // b must stay below 1.0

    // Bucket the inputs by emulated segment. lut_index is inverted, so undo
    // that first: raw_index counts upwards with x.
    std::vector<std::vector<const Point*>> bucket(n_seg);
    for (const Point& p : pts) {
        const int raw_index = bf16_cfg::LUT_MAX_IDX - p.lut_index;
        bucket[raw_index / group].push_back(&p);
    }

    for (int j = 0; j < n_seg; ++j) {
        const double lo = static_cast<double>(j) / n_seg;
        const double hi = static_cast<double>(j + 1) / n_seg;

        // Chebyshev minimax for a linear fit of a convex function: the chord
        // slope, offset halfway between the chord and the parallel tangent.
        const double a0 = (f(lo) - f(hi)) / (hi - lo);
        const double xs = -std::log2(a0 / LN2);              // f'(xs) == -a0
        const double b0 = (f(lo) + a0 * lo + f(xs) + a0 * xs) * 0.5;

        const int64_t aq = std::llround(a0 * static_cast<double>(A_ONE));
        const int64_t bq = std::llround(b0 * static_cast<double>(B_ONE));

        std::vector<int> idxs;
        idxs.reserve(group);
        for (int r = j * group; r < (j + 1) * group; ++r)
            idxs.push_back(bf16_cfg::LUT_MAX_IDX - r);

        // A segment no input ever reaches still needs a sane entry.
        if (bucket[j].empty()) {
            const uint64_t a_fill = static_cast<uint64_t>(std::max<int64_t>(0, std::min<int64_t>(aq, A_ONE - 1)));
            const uint64_t b_fill = static_cast<uint64_t>(std::max<int64_t>(0, std::min<int64_t>(bq, B_ONE - 1)));
            for (int ix : idxs) { tbl.a[ix] = a_fill; tbl.b[ix] = b_fill; }
            continue;
        }

        double   best_ulp   = 1e30;
        int      best_wrong = 1 << 30;
        uint64_t best_a = 0, best_b = 0;
        int64_t  best_da = 0, best_db = 0;
        bool     clipped = false;
        int      radius  = radius0;

        for (;;) {
            for (int64_t da = -radius; da <= radius; ++da) {
                for (int64_t db = -radius; db <= radius; ++db) {
                    // Only the newly added ring needs evaluating on a retry.
                    if (radius > radius0 &&
                        std::abs(da) <= radius / 2 && std::abs(db) <= radius / 2)
                        continue;

                    const int64_t ca = aq + da;
                    const int64_t cb = bq + db;
                    if (ca < 0 || cb < 0) continue;
                    // No integer bit in either coefficient: that is the whole
                    // point of the Q0.17 / Q0.18 format.
                    if (static_cast<uint64_t>(ca) >= A_ONE ||
                        static_cast<uint64_t>(cb) >= B_ONE) { clipped = true; continue; }

                    int          w = 0;
                    const double u = score<CFG>(bucket[j], tbl, idxs,
                                                static_cast<uint64_t>(ca),
                                                static_cast<uint64_t>(cb), w);

                    if (w < best_wrong || (w == best_wrong && u < best_ulp)) {
                        best_wrong = w;  best_ulp = u;
                        best_a = static_cast<uint64_t>(ca);
                        best_b = static_cast<uint64_t>(cb);
                        best_da = da;    best_db = db;
                    }
                }
            }

            const bool on_edge = (std::abs(best_da) == radius ||
                                  std::abs(best_db) == radius);
            if (!on_edge || radius >= radius_max) break;
            radius *= 2;
        }

        for (int ix : idxs) { tbl.a[ix] = best_a; tbl.b[ix] = best_b; }
        if (clipped) ++res.clipped;
        if (radius > res.max_radius) res.max_radius = radius;
    }

    // Final pass over every input with the completed table.
    for (const Point& p : pts) {
        const uint16_t got = pwl_optim::approx<CFG>(p.raw, p.base2, nullptr, &tbl);
        const double   ulp = calculate_ulp_error(p.ref,
                                                 fp_to_double(got, FPType::BF16),
                                                 FPType::BF16);
        if (!std::isfinite(ulp)) continue;

        if (p.base2) {
            if (ulp > res.max_ulp_2) res.max_ulp_2 = ulp;
            if (ulp > 0.5) ++res.wrong_2;
        } else {
            if (ulp > res.max_ulp_e) res.max_ulp_e = ulp;
            if (ulp > 0.5) ++res.wrong_e;
        }
    }
    res.table = tbl;
    return res;
}

/**
 * @brief Emit a synthesisable ROM for an N-entry table.
 *
 * Deliberately matches the layout of the generated
 * src/rtl/bf16_exp2_optim_rom.sv so the two can be synthesised against each
 * other. Only needed to measure what a narrower LUT actually costs: constant
 * contents get folded by synthesis, so a ROM filled with arbitrary values
 * would not give an honest resource number.
 */
template <class CFG>
void emit_rom(int n_seg, const CoeffTable& tbl, const char* rom_style) {
    const int group    = FULL_SEGMENTS / n_seg;
    const int packed_w = CFG::A_F + CFG::B_F;

    std::printf("// Generated by tests/pwl_segment_study.cpp --emit-rom %d\n", n_seg);
    std::printf("// %d entries x %d bits = %d bits. a is Q0.%d, b is Q0.%d,\n",
                n_seg, packed_w, n_seg * packed_w, CFG::A_F, CFG::B_F);
    std::printf("// packed as {b, a}. Measurement aid only -- not the shipped ROM.\n");
    std::printf("`timescale 1ns / 1ps\n\n");
    std::printf("module bf16_exp2_optim_rom #(\n");
    std::printf("    parameter int DATA_W     = %d,\n", packed_w);
    std::printf("    parameter int DEPTH      = %d,\n", n_seg);
    std::printf("    parameter int ADDR_W     = %d,\n",
                static_cast<int>(std::lround(std::log2(n_seg))));
    std::printf("    parameter bit REGISTERED = 1'b1\n");
    std::printf(")(\n");
    std::printf("    input  logic              clk,\n");
    std::printf("    input  logic              pipe_en,\n");
    std::printf("    input  logic [ADDR_W-1:0] addr,\n");
    std::printf("    output logic [DATA_W-1:0] data\n");
    std::printf(");\n\n");
    std::printf("    (* rom_style = \"%s\" *) logic [DATA_W-1:0] rom [0:DEPTH-1];\n\n",
                rom_style);
    std::printf("    initial begin\n");

    // One table entry per emulated segment: inside a group every entry holds
    // the same pair, so the first of each group is representative.
    for (int j = 0; j < n_seg; ++j) {
        const int idx = bf16_cfg::LUT_MAX_IDX - (j * group);
        const uint64_t packed = (tbl.b[idx] << CFG::A_F) | tbl.a[idx];
        std::printf("        rom[%3d] = %d'h%llx;\n", n_seg - 1 - j, packed_w,
                    static_cast<unsigned long long>(packed));
    }

    std::printf("    end\n\n");
    std::printf("    generate\n");
    std::printf("        if (REGISTERED) begin : gen_sync_rom\n");
    std::printf("            always_ff @(posedge clk) begin\n");
    std::printf("                if (pipe_en) data <= rom[addr];\n");
    std::printf("            end\n");
    std::printf("        end else begin : gen_async_rom\n");
    std::printf("            assign data = rom[addr];\n");
    std::printf("        end\n");
    std::printf("    endgenerate\n\n");
    std::printf("endmodule : bf16_exp2_optim_rom\n");
}

/** @brief Run one configuration across the segment counts and print a table. */
template <class CFG>
void sweep(const char* title, const char* note, int radius0, int radius_max) {
    const std::vector<Point> pts = collect_points<CFG>();

    std::printf("\n%s\n%s\n\n", title, note);
    std::printf("|   N | ROM bits | e:maxULP | 2:maxULP | e:miss | 2:miss | search | verdict   |\n");
    std::printf("|----:|---------:|---------:|---------:|-------:|-------:|-------:|-----------|\n");

    for (int n = FULL_SEGMENTS; n >= 4; n /= 2) {
        const FitResult r = fit_and_measure<CFG>(n, pts, radius0, radius_max);
        const bool ok = (r.max_ulp_e <= 0.5 && r.max_ulp_2 <= 0.5);

        std::printf("| %3d | %8d | %8.6f | %8.6f | %6d | %6d | %6d | %-9s |",
                    n, n * 35, r.max_ulp_e, r.max_ulp_2,
                    r.wrong_e, r.wrong_2, r.max_radius, ok ? "<=0.5 ULP" : "FAIL");
        if (r.clipped) std::printf("  (%d segment(s) wanted a coefficient >= 1.0)", r.clipped);
        std::printf("\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    // --emit-rom N [style]: print a synthesisable ROM for an N-entry table
    // and nothing else, so the output can be piped straight to a file.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--emit-rom" && i + 1 < argc) {
            const int   n     = std::atoi(argv[i + 1]);
            const char* style = (i + 2 < argc) ? argv[i + 2] : "block";
            if (n < 1 || FULL_SEGMENTS % n != 0) {
                std::fprintf(stderr, "--emit-rom: N must divide %d\n", FULL_SEGMENTS);
                return 1;
            }
            const std::vector<Point> pts = collect_points<CfgProd>();
            const FitResult r = fit_and_measure<CfgProd>(n, pts, 8, 512);
            std::fprintf(stderr, "emitted %d-entry ROM: e:maxULP=%.6f 2:maxULP=%.6f"
                                 " misses=%d/%d\n",
                         n, r.max_ulp_e, r.max_ulp_2, r.wrong_e, r.wrong_2);
            emit_rom<CfgProd>(n, r.table, style);
            return 0;
        }
    }

    std::printf("PWL segment-count study\n");
    std::printf("=======================\n");
    std::printf("Every segment refitted from scratch at each N: continuous\n");
    std::printf("Chebyshev minimax, then an integer search at the target\n");
    std::printf("quantisation that widens until the winner is off the boundary.\n");

    const std::vector<Point> probe = collect_points<CfgProd>();
    std::printf("\nInputs reaching the polynomial datapath: %zu"
                " (the rest are early-outs and constrain nothing)\n", probe.size());

    sweep<CfgProd>(
        "## Production widths (a Q0.17, b Q0.18, static normaliser)",
        "This is the shipped optim core. 128 is the status quo.", 8, 512);

    sweep<CfgDyn>(
        "## Same widths, priority encoder restored (STATIC_NORM=0)",
        "Isolates the static normaliser's [0.5, 1) assumption from accuracy.", 8, 512);

    sweep<CfgWide>(
        "## Roomier coefficients (a Q0.20, b Q0.21, CALC_F=21)",
        "Isolates segment count from coefficient quantisation.", 8, 512);

    std::printf("\nColumns: miss = results that are not correctly rounded;\n");
    std::printf("search = widest integer neighbourhood any one segment needed.\n");
    std::printf("A row passes only when both modes stay at or below 0.5 ULP.\n");
    return 0;
}

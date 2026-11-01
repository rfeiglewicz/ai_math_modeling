#ifndef BF16_EXP2_PWL_REFIT_HPP
#define BF16_EXP2_PWL_REFIT_HPP

/**
 * @file bf16_exp2_pwl_refit.hpp
 * @brief Refit the PWL coefficient table for an arbitrary segment count and
 *        datapath width, scored on bit-exactness against production.
 *
 * Shared by two programs so that what is studied is exactly what ships:
 *   tests/pwl_width_refit_study.cpp      - sweeps the widths at N segments
 *   modeling/coeff_gen/gen_optim_coeffs.cpp - emits the table for the chosen
 *                                          configuration (C++ header + SV ROM)
 *
 * Why a refit is needed at all: the shipped Q1.20 table has 128 entries and
 * was fitted by gen_bf16_discrete_minmax.py. Narrowing a datapath width
 * against a fixed table measures that table's luck as much as the width
 * (exp_pwl_optim.cpp section 10 shows the non-monotone result: a 17 passes,
 * a 16 fails, a 15 passes). With a fresh fit per configuration the question
 * becomes "does a table exist for these widths", which is what matters.
 *
 * Criterion: every input that reaches the datapath must produce exactly the
 * bits bf16_exp2_approx<29>() produces -- the reference every RTL testbench
 * of the optim core compares against. Additionally b - a*x must stay in
 * [0.5, 1) on every such input, so that the static normaliser (shift by a
 * constant 1) and the guarded one (shift by 1 or 2) agree and the RTL
 * subtractor needs neither sign nor integer bit.
 *
 * The log2(e) product is rounded by the RTL either RNE (ROUND_MODE=0) or
 * half-up (ROUND_MODE=1, the variant used by the retimed builds). Both are
 * verified exhaustively, so the base-e inputs are scored under both: a table
 * that only works for one of them would break the other build.
 *
 * Per segment, the search starts at the continuous Chebyshev minimax line
 * for 2^-x and scans an integer neighbourhood at the target quantisation.
 * The neighbourhood doubles until the winner is off its edge, so the result
 * does not depend on a guessed search width.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "bf16_exp2.hpp"
#include "bf16_exp2_pwl_optim.hpp"
#include "../utils/fp_utils.hpp"

namespace pwl_refit {

/** @brief One (input, mode, log2e-rounding) combination reaching the datapath. */
struct Point {
    uint16_t raw;    ///< BF16 input pattern
    bool     base2;  ///< true = 2^x, false = e^x
    uint16_t want;   ///< bf16_exp2_approx<29>(raw, base2)
    double   ref;    ///< exact result in double, for the tie-break metric
    uint64_t x;      ///< polynomial input, X_F fractional bits
    int32_t  bias;   ///< negated integer part of the shifted value
};

/**
 * @brief The two RTL rounding variants of a width configuration.
 *
 * UNIFIED_F is tied to MANT_MULT_F (the RTL shifter has FRAC_PAD = 0 by
 * construction) and B_F to CALC_F (b feeds the accumulator directly). The
 * normaliser is static: the stricter of the two, see the file header.
 */
template<int MMF, int XF, int CF, int AF>
struct Widths {
    typedef pwl_optim::Cfg<MMF, AC_RND_CONV, MMF, XF, CF, true, AF, CF> Rne;
    typedef pwl_optim::Cfg<MMF, AC_RND,      MMF, XF, CF, true, AF, CF> HalfUp;

    static constexpr int MANT_MULT_F = MMF;
    static constexpr int X_F         = XF;
    static constexpr int CALC_F      = CF;
    static constexpr int A_F         = AF;
    static constexpr int B_F         = CF;
    static constexpr int CDROP       = AF + XF - CF;

    static_assert(CDROP >= 0, "a*x must be at least as fine as the accumulator");
    static_assert(AF + XF <= 62, "a*x must fit in 64 bits");
};

/**
 * @brief Every input that reaches the datapath, under both log2e roundings.
 *
 * @param base_e_only  skip base-2 entirely (expe-only core)
 * @param max_exp      largest input exponent handled by the datapath; the
 *                     base-e-only core early-outs exponent 7 to +0.0
 */
template<class W>
std::vector<Point> collect_points(bool base_e_only, int max_exp) {
    std::vector<Point> pts;
    pts.reserve(6144);

    for (uint32_t i = 0; i <= 0xFFFFu; ++i) {
        const uint16_t raw = static_cast<uint16_t>(i);
        const FPRaw p = fp_decompose(static_cast<uint32_t>(raw), FPType::BF16);

        if (p.status.is_nan || p.status.is_zero || p.status.is_inf) continue;
        if (!p.sign) continue;
        if (p.exponent < bf16_cfg::INPUT_MIN_EXP || p.exponent > max_exp) continue;

        const double xv = fp_to_double(raw, FPType::BF16);

        for (int m = 0; m < 2; ++m) {
            const bool base2 = (m == 1);
            if (base_e_only && base2) continue;

            Point pt;
            pt.raw   = raw;
            pt.base2 = base2;
            pt.want  = bf16_exp2_approx<bf16_cfg::MANT_MULT_F>(raw, base2);
            pt.ref   = base2 ? std::exp2(xv) : std::exp(xv);

            const pwl_optim::FrontEnd fr = pwl_optim::front_end<typename W::Rne>(p, base2);
            pt.x = fr.x;  pt.bias = fr.exponent_bias;
            pts.push_back(pt);

            // Base-2 bypasses the log2e multiply; for base-e the half-up
            // build only adds a point where it actually lands elsewhere.
            if (!base2) {
                const pwl_optim::FrontEnd fh = pwl_optim::front_end<typename W::HalfUp>(p, base2);
                if (fh.x != fr.x || fh.exponent_bias != fr.exponent_bias) {
                    pt.x = fh.x;  pt.bias = fh.exponent_bias;
                    pts.push_back(pt);
                }
            }
        }
    }
    return pts;
}

/** @brief log2 of a power of two. */
inline int ilog2(int n) { int k = 0; while ((1 << k) < n) ++k; return k; }

/** @brief Segment a point falls into, counted upwards with x (not inverted). */
template<class W>
inline int segment_of(const Point& p, int n_seg) {
    return static_cast<int>(p.x >> (W::X_F - ilog2(n_seg)));
}

/**
 * @brief Evaluate the back half of the datapath exactly as the RTL does.
 *
 * res = b - (ax >> CDROP) - (ax[CDROP-1:0] != 0), then normalise and round.
 * @param norm_ok  set false when b - a*x leaves [0.5, 1)
 */
template<class W>
inline uint16_t eval(const Point& p, uint64_t a, uint64_t b, bool& norm_ok) {
    constexpr int CF    = W::CALC_F;
    constexpr int CDROP = W::CDROP;

    const uint64_t ax    = a * p.x;
    const uint64_t ax_hi = ax >> CDROP;
    const uint64_t lo    = (CDROP > 0 && (ax & ((1ull << CDROP) - 1)) != 0) ? 1 : 0;
    int64_t res = static_cast<int64_t>(b) - static_cast<int64_t>(ax_hi) - static_cast<int64_t>(lo);

    norm_ok = (res >= (1ll << (CF - 1))) && (res < (1ll << CF));
    if (res < 0) res = 0;

    FPRaw r = pwl_optim::round_to_bf16<typename W::Rne>(static_cast<pwl_optim::acc_t>(res), p.bias);
    r.sign = false;
    return static_cast<uint16_t>(fp_recompose(r, FPType::BF16));
}

/** @brief Outcome of one refit. */
struct Fit {
    int n_seg = 0;
    std::vector<uint64_t> a;   ///< indexed by segment, counted upwards with x
    std::vector<uint64_t> b;
    int    mismatch_e = 0;     ///< base-e points differing from production
    int    mismatch_2 = 0;     ///< base-2 points differing from production
    int    norm_viol  = 0;     ///< points where b - a*x left [0.5, 1)
    double max_ulp_e  = 0.0;
    double max_ulp_2  = 0.0;
    int    max_radius = 0;     ///< widest search any segment needed
    uint64_t a_min = ~0ull, a_max = 0, b_min = ~0ull, b_max = 0;

    bool ok() const { return mismatch_e == 0 && mismatch_2 == 0 && norm_viol == 0; }
};

/** @brief Score one candidate on one segment: (failures, worst ULP). */
template<class W>
inline void score(const std::vector<const Point*>& grp, uint64_t a, uint64_t b,
                  int& fails, double& worst) {
    fails = 0;
    worst = 0.0;
    for (const Point* p : grp) {
        bool nok = true;
        const uint16_t got = eval<W>(*p, a, b, nok);
        if (got != p->want || !nok) ++fails;
        const double u = calculate_ulp_error(p->ref, fp_to_double(got, FPType::BF16),
                                             FPType::BF16);
        if (std::isfinite(u) && u > worst) worst = u;
    }
}

/**
 * @brief Refit an @p n_seg entry table for widths @p W over @p pts.
 *
 * @param radius0    initial half-width of the integer search
 * @param radius_max point at which to stop doubling and accept the result
 */
template<class W>
Fit fit(int n_seg, const std::vector<Point>& pts, int radius0 = 8, int radius_max = 256) {
    Fit res;
    res.n_seg = n_seg;
    res.a.assign(n_seg, 0);
    res.b.assign(n_seg, 0);

    const int64_t A_ONE = 1ll << W::A_F;
    const int64_t B_ONE = 1ll << W::B_F;

    std::vector<std::vector<const Point*>> bucket(n_seg);
    for (const Point& p : pts) bucket[segment_of<W>(p, n_seg)].push_back(&p);

    auto f = [](double x) { return std::exp2(-x); };
    const double LN2 = 0.69314718055994530942;

    for (int j = 0; j < n_seg; ++j) {
        const double lo = static_cast<double>(j) / n_seg;
        const double hi = static_cast<double>(j + 1) / n_seg;

        // Chebyshev minimax line for the convex 2^-x: chord slope, offset
        // halfway between the chord and the parallel tangent.
        const double a0 = (f(lo) - f(hi)) / (hi - lo);
        const double xs = -std::log2(a0 / LN2);
        const double b0 = (f(lo) + a0 * lo + f(xs) + a0 * xs) * 0.5;

        const int64_t aq = std::llround(a0 * static_cast<double>(A_ONE));
        const int64_t bq = std::llround(b0 * static_cast<double>(B_ONE));

        if (bucket[j].empty()) {
            res.a[j] = static_cast<uint64_t>(std::max<int64_t>(0, std::min<int64_t>(aq, A_ONE - 1)));
            res.b[j] = static_cast<uint64_t>(std::max<int64_t>(0, std::min<int64_t>(bq, B_ONE - 1)));
            continue;
        }

        int      best_fail = 1 << 30;
        double   best_ulp  = 1e30;
        uint64_t best_a = 0, best_b = 0;
        int64_t  best_da = 0, best_db = 0;
        int      radius = radius0;

        for (;;) {
            for (int64_t da = -radius; da <= radius; ++da) {
                for (int64_t db = -radius; db <= radius; ++db) {
                    if (radius > radius0 &&
                        std::abs(da) <= radius / 2 && std::abs(db) <= radius / 2)
                        continue;   // already scored on the previous ring

                    const int64_t ca = aq + da;
                    const int64_t cb = bq + db;
                    if (ca < 0 || cb < 0 || ca >= A_ONE || cb >= B_ONE) continue;

                    int fails; double u;
                    score<W>(bucket[j], static_cast<uint64_t>(ca), static_cast<uint64_t>(cb), fails, u);
                    if (fails < best_fail || (fails == best_fail && u < best_ulp)) {
                        best_fail = fails;  best_ulp = u;
                        best_a = static_cast<uint64_t>(ca);
                        best_b = static_cast<uint64_t>(cb);
                        best_da = da;  best_db = db;
                    }
                }
            }
            const bool on_edge = (std::abs(best_da) == radius || std::abs(best_db) == radius);
            if (!on_edge || radius >= radius_max) break;
            radius *= 2;
        }

        res.a[j] = best_a;
        res.b[j] = best_b;
        if (radius > res.max_radius) res.max_radius = radius;
    }

    // Final pass with the completed table.
    for (const Point& p : pts) {
        const int j = segment_of<W>(p, n_seg);
        bool nok = true;
        const uint16_t got = eval<W>(p, res.a[j], res.b[j], nok);
        if (!nok) ++res.norm_viol;
        if (got != p.want) { if (p.base2) ++res.mismatch_2; else ++res.mismatch_e; }

        const double u = calculate_ulp_error(p.ref, fp_to_double(got, FPType::BF16), FPType::BF16);
        if (std::isfinite(u)) {
            if (p.base2) res.max_ulp_2 = std::max(res.max_ulp_2, u);
            else         res.max_ulp_e = std::max(res.max_ulp_e, u);
        }
    }
    for (int j = 0; j < n_seg; ++j) {
        if (bucket[j].empty()) continue;
        res.a_min = std::min(res.a_min, res.a[j]);  res.a_max = std::max(res.a_max, res.a[j]);
        res.b_min = std::min(res.b_min, res.b[j]);  res.b_max = std::max(res.b_max, res.b[j]);
    }
    return res;
}

/**
 * @brief Independent cross-check of a fitted table through the full model.
 *
 * Expands the N-entry table onto pwl_optim's 128-entry CoeffTable (each
 * entry repeated 128/N times, inverted index) and runs pwl_optim::approx on
 * every BF16 pattern under both log2e roundings. Catches any disagreement
 * between eval() above and the reference model, including the early-out
 * ladder. Returns the number of mismatching (pattern, mode, rounding) cases.
 */
template<class W>
int cross_check(const Fit& f, bool base_e_only, int max_exp) {
    pwl_optim::CoeffTable tbl;
    const int group = bf16_cfg::LUT_SIZE / f.n_seg;
    for (int i = 0; i < bf16_cfg::LUT_SIZE; ++i) {
        const int raw7 = bf16_cfg::LUT_MAX_IDX - i;
        tbl.a[i] = f.a[raw7 / group];
        tbl.b[i] = f.b[raw7 / group];
    }

    int bad = 0;
    for (uint32_t i = 0; i <= 0xFFFFu; ++i) {
        const uint16_t raw = static_cast<uint16_t>(i);
        const FPRaw p = fp_decompose(static_cast<uint32_t>(raw), FPType::BF16);
        // The base-e-only core early-outs exponents above max_exp to +0.0;
        // pwl_optim::approx would send them through the datapath instead.
        const bool early_zero = !p.status.is_nan && !p.status.is_zero && !p.status.is_inf
                              && p.sign && p.exponent > max_exp;

        for (int m = 0; m < 2; ++m) {
            const bool base2 = (m == 1);
            if (base_e_only && base2) continue;
            const uint16_t want = bf16_exp2_approx<bf16_cfg::MANT_MULT_F>(raw, base2);
            const uint16_t g1 = early_zero ? 0 : pwl_optim::approx<typename W::Rne>(raw, base2, nullptr, &tbl);
            const uint16_t g2 = early_zero ? 0 : pwl_optim::approx<typename W::HalfUp>(raw, base2, nullptr, &tbl);
            if (g1 != want) ++bad;
            if (g2 != want) ++bad;
        }
    }
    return bad;
}

} // namespace pwl_refit

#endif // BF16_EXP2_PWL_REFIT_HPP

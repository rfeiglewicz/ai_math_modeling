/**
 * @file pwl_width_refit_study.cpp
 * @brief Minimum datapath widths of the optim PWL core at N segments, with
 *        the coefficient table refitted for every configuration.
 *
 * tests/exp_pwl_optim.cpp found the current widths against the shipped
 * 128-entry table. tests/pwl_segment_study.cpp then showed that 64 segments
 * are enough at those widths. This study closes the loop: at 64 segments,
 * how narrow can each width go when the table is allowed to follow?
 *
 * Every row is a complete refit (bf16_exp2_pwl_refit.hpp) followed by an
 * exhaustive check. Pass = every input reaching the datapath gives exactly
 * the bits of bf16_exp2_approx<29>(), under both RTL log2e roundings (RNE
 * and half-up), with b - a*x inside [0.5, 1).
 *
 * Two criteria are run:
 *   both modes  - the shipped dual-mode core, exponents [-9, 7]
 *   base-e only - the BASE_E_ONLY=1 core, exponents [-9, 6]
 *
 * Widths are swept one axis at a time from the current configuration and
 * reported in full, not bisected: earlier sweeps were non-monotone.
 *
 * Build: make pwl_width_refit_study
 */

#include <cstdio>
#include <cstring>
#include <string>

#include "bf16_exp2_pwl_refit.hpp"

using pwl_refit::Widths;

namespace {

int g_nseg = 64;

template<class W>
pwl_refit::Fit row(const char* label, bool base_e_only, bool xcheck = false) {
    const int max_exp = base_e_only ? 6 : 7;
    const std::vector<pwl_refit::Point> pts = pwl_refit::collect_points<W>(base_e_only, max_exp);
    const pwl_refit::Fit f = pwl_refit::fit<W>(g_nseg, pts);

    std::printf("| %-14s | %3d | %3d | %3d | %3d | %3d | %5d | %4d | %4d | %4d | %8.6f | %8.6f | %4d | %-4s |",
                label, W::MANT_MULT_F, W::X_F, W::CALC_F, W::A_F, W::B_F,
                g_nseg * (W::A_F + W::B_F),
                f.mismatch_e, f.mismatch_2, f.norm_viol,
                f.max_ulp_e, f.max_ulp_2, f.max_radius, f.ok() ? "PASS" : "fail");
    if (xcheck) {
        const int bad = pwl_refit::cross_check<W>(f, base_e_only, max_exp);
        std::printf(" xcheck %s", bad == 0 ? "ok" : "MISMATCH");
        if (bad) std::printf(" (%d)", bad);
    }
    std::printf("\n");
    std::fflush(stdout);
    return f;
}

void head(const char* title) {
    std::printf("\n%s\n\n", title);
    std::printf("| %-14s | MMF |  XF |  CF |  AF |  BF | ROM b | e:mm | 2:mm | norm | e:maxULP | 2:maxULP | srch | res  |\n", "config");
    std::printf("|----------------|----:|----:|----:|----:|----:|------:|-----:|-----:|-----:|---------:|---------:|-----:|------|\n");
}

template<int MMF> void sweep_mmf(bool e) {
    row<Widths<MMF, 17, 18, 17>>("MMF", e);
    if constexpr (MMF > 17) sweep_mmf<MMF - 1>(e);   // X_F <= UNIFIED_F = MMF
}
template<int XF> void sweep_xf(bool e) {
    row<Widths<21, XF, 18, 17>>("XF", e);
    if constexpr (XF > 12) sweep_xf<XF - 1>(e);
}
template<int CF> void sweep_cf(bool e) {
    row<Widths<21, 17, CF, 17>>("CF = BF", e);
    if constexpr (CF > 14) sweep_cf<CF - 1>(e);
}
template<int AF> void sweep_af(bool e) {
    row<Widths<21, 17, 18, AF>>("AF", e);
    if constexpr (AF > 10) sweep_af<AF - 1>(e);
}

void study(bool e) {
    const char* m = e ? "base-e only, exponents [-9, 6]" : "both modes, exponents [-9, 7]";
    std::printf("\n## %d segments, %s\n", g_nseg, m);

    head("### Current widths (MMF 21, XF 17, CF = BF 18, AF 17)");
    row<Widths<21, 17, 18, 17>>("current", e, true);

    head("### log2(e) product / unified shift fraction (UF = MMF)");
    sweep_mmf<21>(e);
    head("### polynomial input x");
    sweep_xf<17>(e);
    head("### accumulator = offset b");
    sweep_cf<18>(e);
    head("### slope a");
    sweep_af<17>(e);
}

// --- Joint combinations of the per-axis survivors -----------------------
// The axes interact (a narrower x shifts where the accumulator's truncation
// lands, a narrower a changes which (a, b) pairs exist), so the per-axis
// minima are only candidates until they have been checked together.

template<int MMF, int XF, int CF, int AF>
void joint_cell(bool e) {
    if constexpr (XF <= MMF) row<Widths<MMF, XF, CF, AF>>("joint", e, true);
}
template<int MMF, int XF, int CF>
void joint_af(bool e) {
    joint_cell<MMF, XF, CF, 14>(e);
    joint_cell<MMF, XF, CF, 13>(e);
    joint_cell<MMF, XF, CF, 12>(e);
}
template<int MMF, int XF>
void joint_cf(bool e) { joint_af<MMF, XF, 18>(e); joint_af<MMF, XF, 17>(e); }
template<int MMF>
void joint_xf(bool e) { joint_cf<MMF, 17>(e); joint_cf<MMF, 16>(e); }

void joint(bool e) {
    const char* m = e ? "base-e only, exponents [-9, 6]" : "both modes, exponents [-9, 7]";
    std::printf("\n## %d segments, %s: joint combinations\n", g_nseg, m);
    head("### MMF in {21, 19, 18} x XF in {17, 16} x CF in {18, 17} x AF in {14, 13, 12}");
    joint_xf<21>(e);
    joint_xf<19>(e);
    joint_xf<18>(e);

    // Every corner of the grid passes, so the narrowest one is the candidate.
    // Check that one step past it on any axis really does fail, with the
    // other three held at the candidate -- otherwise the grid was too small.
    head("### Candidate (MMF 18, XF 16, CF = BF 17, AF 12) and one step past it");
    row<Widths<18, 16, 17, 12>>("candidate", e, true);
    row<Widths<17, 16, 17, 12>>("MMF - 1", e, true);
    row<Widths<16, 16, 17, 12>>("MMF - 2", e, true);
    row<Widths<18, 15, 17, 12>>("XF - 1", e, true);
    row<Widths<18, 16, 16, 12>>("CF - 1", e, true);
    row<Widths<18, 16, 17, 11>>("AF - 1", e, true);
}

// --- Full grid ------------------------------------------------------------
// The one-step checks above showed the landscape is not monotone even with a
// refit (CF 16 passes jointly although it failed on its own axis), so the
// candidate is only trustworthy once the whole neighbourhood is enumerated.

struct GridResult { int mmf, xf, cf, af; bool both, e_only; };
std::vector<GridResult> g_grid;

template<int MMF, int XF, int CF, int AF>
void grid_cell() {
    if constexpr (XF <= MMF && AF + XF >= CF) {
        GridResult g{MMF, XF, CF, AF, false, false};
        typedef Widths<MMF, XF, CF, AF> W;
        g.both   = pwl_refit::fit<W>(g_nseg, pwl_refit::collect_points<W>(false, 7)).ok();
        g.e_only = pwl_refit::fit<W>(g_nseg, pwl_refit::collect_points<W>(true, 6)).ok();
        g_grid.push_back(g);
    }
}
template<int MMF, int XF, int CF, int AF> void grid_af() {
    grid_cell<MMF, XF, CF, AF>();
    if constexpr (AF > 10) grid_af<MMF, XF, CF, AF - 1>();
}
template<int MMF, int XF, int CF> void grid_cf() {
    grid_af<MMF, XF, CF, 14>();
    if constexpr (CF > 15) grid_cf<MMF, XF, CF - 1>();
}
template<int MMF, int XF> void grid_xf() {
    grid_cf<MMF, XF, 18>();
    if constexpr (XF > 14) grid_xf<MMF, XF - 1>();
}
template<int MMF> void grid_mmf() {
    grid_xf<MMF, 17>();
    if constexpr (MMF > 16) grid_mmf<MMF - 1>();
}

void grid() {
    std::printf("\n## %d segments: full grid MMF 21..16, XF 17..14, CF 18..15, AF 14..10\n\n", g_nseg);
    grid_mmf<21>();

    int n_both = 0, n_e = 0;
    for (const GridResult& g : g_grid) { n_both += g.both; n_e += g.e_only; }
    std::printf("configurations: %zu, pass both modes: %d, pass base-e only: %d\n\n",
                g_grid.size(), n_both, n_e);

    // Narrowest first, by the sum of the four fabric-relevant widths.
    std::vector<GridResult> s = g_grid;
    std::stable_sort(s.begin(), s.end(), [](const GridResult& a, const GridResult& b) {
        return (a.mmf + a.xf + a.cf + a.af) < (b.mmf + b.xf + b.cf + b.af);
    });
    std::printf("| MMF |  XF |  CF |  AF | sum | ROM b | both | e-only |\n");
    std::printf("|----:|----:|----:|----:|----:|------:|------|--------|\n");
    int shown = 0;
    for (const GridResult& g : s) {
        if (!g.both && !g.e_only) continue;
        std::printf("| %3d | %3d | %3d | %3d | %3d | %5d | %-4s | %-6s |\n",
                    g.mmf, g.xf, g.cf, g.af, g.mmf + g.xf + g.cf + g.af,
                    g_nseg * (g.af + g.cf), g.both ? "PASS" : "-", g.e_only ? "PASS" : "-");
        if (++shown >= 40) break;
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool only_e = false, only_both = false, do_joint = false, do_grid = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--segments" && i + 1 < argc) g_nseg = std::atoi(argv[++i]);
        else if (a == "--base-e-only") only_e = true;
        else if (a == "--both-only")   only_both = true;
        else if (a == "--joint")       do_joint = true;
        else if (a == "--grid")        do_grid = true;
        else { std::fprintf(stderr, "usage: %s [--segments N] [--base-e-only|--both-only] [--joint]\n", argv[0]); return 2; }
    }
    if (g_nseg < 2 || g_nseg > bf16_cfg::LUT_SIZE || (g_nseg & (g_nseg - 1))) {
        std::fprintf(stderr, "--segments must be a power of two in [2, %d]\n", bf16_cfg::LUT_SIZE);
        return 2;
    }

    std::printf("PWL width study with per-configuration refit\n");
    std::printf("============================================\n");
    std::printf("Pass = bit-exact vs bf16_exp2_approx<29> under RNE and half-up log2e\n");
    std::printf("rounding, and b - a*x in [0.5, 1) on every input reaching the datapath.\n");
    std::printf("Columns: mm = mismatching points, norm = [0.5, 1) violations,\n");
    std::printf("srch = widest integer neighbourhood any one segment needed.\n");

    if (do_grid) { grid(); return 0; }
    if (do_joint) {
        if (!only_e)    joint(false);
        if (!only_both) joint(true);
        return 0;
    }
    if (!only_e)    study(false);
    if (!only_both) study(true);
    return 0;
}

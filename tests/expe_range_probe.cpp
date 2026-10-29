/**
 * @file expe_range_probe.cpp
 * @brief Measures the input exponent range the base-e core actually needs.
 *
 * bf16_early_out clamps on INPUT_MIN_EXP / INPUT_MAX_EXP, which are currently
 * -9 and 7. Those bounds are shared by base-2 and base-e, so they are the
 * UNION of what the two modes need. This probe asks what each mode needs on
 * its own, by finding the extreme input exponents for which the production
 * model returns something other than the early-out constants.
 *
 * The answer sets IN_CONV_INT_W for a base-e-only core: a narrower
 * INPUT_MAX_EXP means a narrower unified shift register.
 *
 * Build:  make expe_range_probe
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>

#include "../src/approximations/bf16_exp2.hpp"
#include "fp_utils.hpp"

namespace {

struct Range {
    int  min_exp   = 999;   // smallest exponent needing the core
    int  max_exp   = -999;  // largest exponent needing the core
    long core_cnt  = 0;     // patterns routed to the polynomial core
    long one_cnt   = 0;     // patterns returning exactly +1.0
    long zero_cnt  = 0;     // patterns returning exactly +0.0
};

/**
 * @brief Classify every negative finite BF16 input for one mode.
 *
 * A pattern "needs the core" when the production result is neither +1.0
 * (0x3F80) nor +0.0 (0x0000). Those two are exactly what bf16_early_out can
 * produce without the datapath, so anything else must come from the core.
 */
Range probe(bool base2) {
    Range r;

    for (uint32_t i = 0; i <= 0xFFFFu; ++i) {
        const uint16_t in = static_cast<uint16_t>(i);
        const double   x  = fp_to_double(in, FPType::BF16);

        // The core path only ever sees negative finite inputs; everything
        // else is an early-out by construction.
        if (!std::isfinite(x) || x >= 0.0) continue;

        const uint16_t out = bf16_exp2_approx(in, base2);

        if (out == 0x3F80u)      { r.one_cnt++;  continue; }
        if (out == 0x0000u)      { r.zero_cnt++; continue; }

        // Unbiased exponent of the input.
        const int biased = static_cast<int>((in >> 7) & 0xFFu);
        const int exp    = biased - 127;

        if (exp < r.min_exp) r.min_exp = exp;
        if (exp > r.max_exp) r.max_exp = exp;
        r.core_cnt++;
    }
    return r;
}

void report(const char* tag, const Range& r) {
    std::cout << "  " << std::left << std::setw(8) << tag << std::right
              << "  core path needed for exp in [" << std::setw(3) << r.min_exp
              << ", " << std::setw(2) << r.max_exp << "]"
              << "   patterns: core=" << std::setw(5) << r.core_cnt
              << " one=" << std::setw(5) << r.one_cnt
              << " zero=" << std::setw(5) << r.zero_cnt << "\n";
}

} // namespace

int main() {
    std::cout << "\n=== Input exponent range actually required, per mode ===\n\n";
    std::cout << "  Current shared bounds: INPUT_MIN_EXP=" << bf16_cfg::INPUT_MIN_EXP
              << "  INPUT_MAX_EXP=" << bf16_cfg::INPUT_MAX_EXP << "\n\n";

    const Range e2 = probe(true);
    const Range ee = probe(false);

    report("base-2", e2);
    report("base-e", ee);

    std::cout << "\n  Union (what the shared bounds must cover): ["
              << std::min(e2.min_exp, ee.min_exp) << ", "
              << std::max(e2.max_exp, ee.max_exp) << "]\n";

    std::cout << "\n=== Consequence for a base-e-only core ===\n\n";

    const int mant_mult_i = bf16_cfg::MANT_MULT_I;
    const int shared_w    = bf16_cfg::INPUT_MAX_EXP + mant_mult_i;
    const int expe_w      = ee.max_exp + mant_mult_i;

    std::cout << "  IN_CONV_INT_W = INPUT_MAX_EXP + MANT_MULT_I\n"
              << "    shared  : " << bf16_cfg::INPUT_MAX_EXP << " + " << mant_mult_i
              << " = " << shared_w << " bits\n"
              << "    base-e  : " << ee.max_exp << " + " << mant_mult_i
              << " = " << expe_w << " bits"
              << (expe_w < shared_w ? "   <-- one bit narrower\n" : "   (no change)\n");

    std::cout << "\n  Sanity: the largest |x| that does not flush to zero.\n";
    for (int exp = ee.max_exp; exp <= bf16_cfg::INPUT_MAX_EXP + 1; ++exp) {
        const double lo = std::ldexp(1.0, exp);
        std::cout << "    exp=" << std::setw(2) << exp
                  << "  |x| in [" << std::setw(6) << lo << ", " << std::setw(6) << 2.0 * lo << ")"
                  << "   e^-|x| <= 2^" << std::fixed << std::setprecision(1)
                  << (-lo * 1.4426950408889634) << "\n";
    }
    std::cout << "    BF16 flushes to zero below 2^-133 (min denormal).\n";

    std::cout << "\n";
    return 0;
}

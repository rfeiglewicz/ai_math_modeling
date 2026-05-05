/**
 * @file rne_rounding_sweep.cpp
 * @brief Sweep test: find the maximum bit reduction of mant_mult (2.29)
 *        after log2(e) multiplication that still keeps ≤ 0.5 ULP error.
 *
 * For each MANT_MULT_ROUND_FRAC value from 29 (full) down to 1, the test
 * runs the C++ model for all 32768 negative BF16 inputs in base-e mode,
 * computes ULP error vs double-precision std::exp(), and reports max/avg ULP.
 *
 * Build:
 *   make rne_sweep
 * Run (from project root):
 *   ./build/rne_rounding_sweep
 */

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <limits>
#include "../src/approximations/bf16_exp2.hpp"
#include "fp_utils.hpp"

// ── per-width analysis ──────────────────────────────────────────────────

struct SweepResult {
    int     frac_bits;      // MANT_MULT_ROUND_FRAC value
    int     total_bits;     // frac_bits + 2 (integer bits)
    double  max_ulp;        // worst-case ULP error
    uint16_t max_ulp_input; // BF16 input that produced max ULP
    double  avg_ulp;        // average ULP across valid measurements
    int     valid_count;    // number of finite-error samples
};

template<int FRAC>
SweepResult run_sweep_for_width() {
    SweepResult res{};
    res.frac_bits  = FRAC;
    res.total_bits = FRAC + bf16_cfg::MANT_MULT_I;
    res.max_ulp    = 0.0;
    res.avg_ulp    = 0.0;
    res.valid_count = 0;

    double total_ulp = 0.0;

    for (uint32_t i = 0x8000; i <= 0xFFFF; ++i) {
        uint16_t input_raw = static_cast<uint16_t>(i);

        // Run base-e approximation with rounded mant_mult
        uint16_t output_raw = bf16_exp2_approx<FRAC>(input_raw, /*base2=*/false);

        // Reference value
        double x_val      = fp_to_double(input_raw, FPType::BF16);
        double approx_val = fp_to_double(output_raw, FPType::BF16);
        double reference  = std::exp(x_val);

        double ulp_err = calculate_ulp_error(reference, approx_val, FPType::BF16);

        if (std::isfinite(ulp_err)) {
            if (ulp_err > res.max_ulp) {
                res.max_ulp      = ulp_err;
                res.max_ulp_input = input_raw;
            }
            total_ulp += ulp_err;
            res.valid_count++;
        }
    }

    if (res.valid_count > 0) {
        res.avg_ulp = total_ulp / res.valid_count;
    }

    return res;
}

// ── Also verify base2 is unaffected ─────────────────────────────────────

template<int FRAC>
double check_base2_max_ulp() {
    double max_ulp = 0.0;
    for (uint32_t i = 0x8000; i <= 0xFFFF; ++i) {
        uint16_t input_raw = static_cast<uint16_t>(i);
        uint16_t output_raw = bf16_exp2_approx<FRAC>(input_raw, /*base2=*/true);

        double x_val      = fp_to_double(input_raw, FPType::BF16);
        double approx_val = fp_to_double(output_raw, FPType::BF16);
        double reference  = std::exp2(x_val);

        double ulp_err = calculate_ulp_error(reference, approx_val, FPType::BF16);
        if (std::isfinite(ulp_err) && ulp_err > max_ulp) {
            max_ulp = ulp_err;
        }
    }
    return max_ulp;
}

// ── Recursive template instantiation to sweep widths ────────────────────

template<int FRAC>
void sweep_recursive(SweepResult results[], int& idx) {
    results[idx++] = run_sweep_for_width<FRAC>();
    if constexpr (FRAC > 1) {
        sweep_recursive<FRAC - 1>(results, idx);
    }
}

// ── main ────────────────────────────────────────────────────────────────

int main() {
    constexpr int MAX_FRAC = bf16_cfg::MANT_MULT_F;  // 29
    SweepResult results[MAX_FRAC];  // indices 0..28 → frac 29..1
    int idx = 0;

    std::cout << "=== RNE Rounding Sweep after log2(e) Multiply (base-e) ===\n";
    std::cout << "Full precision: 2." << MAX_FRAC
              << " (" << (MAX_FRAC + bf16_cfg::MANT_MULT_I) << " bits total)\n";
    std::cout << "Sweeping MANT_MULT_ROUND_FRAC from " << MAX_FRAC << " down to 1 ...\n\n";

    sweep_recursive<MAX_FRAC>(results, idx);

    // ── Print table ──
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "+----------+-------------+-----------+-----------+--------+--------+\n";
    std::cout << "| Frac (F) | Format      | Total [b] | Max ULP   | Avg ULP| Status |\n";
    std::cout << "+----------+-------------+-----------+-----------+--------+--------+\n";

    int best_frac = MAX_FRAC;
    for (int i = 0; i < idx; ++i) {
        const auto& r = results[i];
        const char* status = (r.max_ulp <= 0.5) ? " PASS " : " FAIL ";
        std::cout << "| " << std::setw(8) << r.frac_bits
                  << " | 2." << std::setw(2) << r.frac_bits
                  << std::setw(7) << " "
                  << " | " << std::setw(9) << r.total_bits
                  << " | " << std::setw(9) << r.max_ulp
                  << " | " << std::setw(6) << r.avg_ulp
                  << " |" << status << "|\n";

        if (r.max_ulp <= 0.5 && r.frac_bits < best_frac) {
            best_frac = r.frac_bits;
        }
    }
    std::cout << "+----------+-------------+-----------+-----------+--------+--------+\n\n";

    // ── Summary ──
    if (best_frac < MAX_FRAC) {
        int saved = MAX_FRAC - best_frac;
        std::cout << ">> Maximum bit reduction while keeping <= 0.5 ULP:\n";
        std::cout << "   MANT_MULT_ROUND_FRAC = " << best_frac
                  << "  (format 2." << best_frac
                  << ", " << (best_frac + bf16_cfg::MANT_MULT_I) << " bits total)\n";
        std::cout << "   Bits saved: " << saved
                  << " (from " << (MAX_FRAC + bf16_cfg::MANT_MULT_I)
                  << " down to " << (best_frac + bf16_cfg::MANT_MULT_I) << ")\n";
    } else {
        std::cout << ">> No bit reduction possible while keeping <= 0.5 ULP.\n";
    }

    // ── Verify base2 mode is unaffected with minimum width ──
    std::cout << "\n--- Verifying base2 mode is unaffected (FRAC=" << best_frac << ") ---\n";
    // We call with FRAC=1 (worst case) to show base2 never uses mant_mult
    double base2_max_ulp = check_base2_max_ulp<1>();
    std::cout << "Base2 max ULP (FRAC=1): " << base2_max_ulp;
    if (base2_max_ulp <= 0.5) {
        std::cout << " [OK — base2 path unaffected]\n";
    } else {
        std::cout << " [UNEXPECTED — base2 should not depend on mant_mult rounding]\n";
    }

    return 0;
}

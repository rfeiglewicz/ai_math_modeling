// =============================================================================
// bf16_expe_lut_test.cpp
// Verifies that the full lookup-table exp(x) model (bf16_expe_lut_approx)
// achieves <= 0.5 ULP over its tabulated domain: negative BF16 inputs whose
// unbiased exponent lies in [-9, 7].
//
// Also runs a full 65536-value sweep, reporting the maximum ULP within the
// tabulated domain (special/clamped inputs are excluded from the assertion
// since the model intentionally returns 1.0 / 0.0 there).
// =============================================================================

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <limits>
#include "bf16_expe_lut.hpp"
#include "fp_utils.hpp"

int main() {
    std::cout << "=== BF16 expe FULL-LUT model: ULP verification ===\n";
    std::cout << "Tabulated domain: negative inputs, unbiased exponent in ["
              << bf16_expe_lut::LUT_MIN_EXP << ", "
              << bf16_expe_lut::LUT_MAX_EXP << "]\n";
    std::cout << "Table size: " << bf16_expe_lut::LUT_SIZE << " entries\n\n";

    double   max_ulp        = 0.0;
    uint16_t max_ulp_input  = 0;
    double   total_ulp      = 0.0;
    uint64_t checked        = 0;
    uint64_t over_half      = 0;

    // Sweep the exact tabulated domain.
    for (int e = bf16_expe_lut::LUT_MIN_EXP; e <= bf16_expe_lut::LUT_MAX_EXP; ++e) {
        for (int m = 0; m < bf16_expe_lut::LUT_MANT_COUNT; ++m) {
            uint32_t biased_exp = static_cast<uint32_t>(e + 127);
            uint16_t raw_input  = static_cast<uint16_t>(
                (1u << 15) | (biased_exp << 7) | static_cast<uint32_t>(m));

            uint16_t out_bf16 = bf16_expe_lut_approx(raw_input);

            double x          = fp_to_double(raw_input, FPType::BF16);
            double approx_val = fp_to_double(out_bf16, FPType::BF16);
            double reference  = std::exp(x);

            double ulp = calculate_ulp_error(reference, approx_val, FPType::BF16);

            if (std::isfinite(ulp)) {
                if (ulp > max_ulp) {
                    max_ulp       = ulp;
                    max_ulp_input = raw_input;
                }
                total_ulp += ulp;
                if (ulp > 0.5) ++over_half;
            }
            ++checked;
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Checked entries : " << checked << "\n";
    std::cout << "Max ULP error   : " << max_ulp
              << " (at input 0x" << std::hex << std::uppercase << std::setw(4)
              << std::setfill('0') << max_ulp_input << std::dec
              << std::nouppercase << std::setfill(' ') << ")\n";
    std::cout << "Average ULP     : " << (checked ? total_ulp / checked : 0.0) << "\n";
    std::cout << "Entries > 0.5   : " << over_half << "\n\n";

    // -------------------------------------------------------------------------
    // Full 65536-value sweep -> write "INPUT OUTPUT ULP" file (matches the
    // golden_ref/output_with_ulp.txt format: 4-digit uppercase hex + 4-dp ULP).
    // -------------------------------------------------------------------------
    const std::string out_path = "modeling/golden_ref/bf16_expe_lut_output_with_ulp.txt";
    std::ofstream ofs(out_path);
    if (!ofs.is_open()) {
        std::cerr << "Error: cannot open " << out_path << " for writing.\n";
        return 2;
    }

    for (uint32_t i = 0; i <= 0xFFFF; ++i) {
        uint16_t raw_input = static_cast<uint16_t>(i);
        uint16_t out_bf16  = bf16_expe_lut_approx(raw_input);

        double x          = fp_to_double(raw_input, FPType::BF16);
        double approx_val = fp_to_double(out_bf16, FPType::BF16);
        double reference  = std::exp(x);

        double ulp = calculate_ulp_error(reference, approx_val, FPType::BF16);
        if (!std::isfinite(ulp)) ulp = 0.0; // NaN/Inf cases -> report 0.0

        ofs << std::hex << std::uppercase << std::setfill('0')
            << std::setw(4) << raw_input << ' '
            << std::setw(4) << out_bf16 << ' '
            << std::dec << std::nouppercase << std::setfill(' ')
            << std::fixed << std::setprecision(4) << ulp << '\n';
    }
    ofs.close();
    std::cout << "Full-sweep ULP file written to: " << out_path << "\n\n";

    bool pass = (max_ulp <= 0.5) && (over_half == 0);
    std::cout << "RESULT: " << (pass ? "PASS (<= 0.5 ULP)" : "FAIL (> 0.5 ULP)") << "\n";

    return pass ? 0 : 1;
}

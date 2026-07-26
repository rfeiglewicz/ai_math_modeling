// Exhaustive equivalence test: cut-point ladder model vs the full LUT model.
//
// The full LUT stores RNE-rounded exp(x) for every BF16 input, so bit-exact
// agreement over all 65536 payloads proves the ladder is <= 0.5 ULP.

#include "bf16_expe_cut.hpp"
#include "bf16_expe_lut.hpp"
#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    uint64_t mismatches = 0;
    uint64_t checked = 0;

    for (uint32_t raw = 0; raw <= 0xFFFF; ++raw) {
        const uint16_t input = static_cast<uint16_t>(raw);
        const uint16_t expected = bf16_expe_lut_approx(input);
        const uint16_t actual = bf16_expe_cut_approx(input);
        ++checked;

        if (actual != expected) {
            ++mismatches;
            if (mismatches <= 20) {
                std::cout << "Mismatch input=0x" << std::hex << std::uppercase
                          << std::setw(4) << std::setfill('0') << input
                          << " expected=0x" << std::setw(4) << expected
                          << " actual=0x" << std::setw(4) << actual
                          << std::dec << std::setfill(' ') << '\n';
            }
        }
    }

    constexpr int full_lut_bits = 2176 * 16;
    constexpr int hybrid_bits = 1024 * 16 + 98 * 7 + 8 * 16;
    constexpr int cut_bits = bf16_expe_cut::PAYLOAD_BITS;

    std::cout << "\nInputs checked    : " << checked << '\n'
              << "Mismatches        : " << mismatches << "\n\n"
              << "Full LUT payload  : " << std::setw(6) << full_lut_bits << " bits\n"
              << "Hybrid payload    : " << std::setw(6) << hybrid_bits << " bits\n"
              << "Cut ladder payload: " << std::setw(6) << cut_bits << " bits\n"
              << "  cut ROM         : " << std::setw(6)
              << bf16_expe_cut::CUT_COUNT * bf16_expe_cut::CUT_DEV_BITS << " bits ("
              << bf16_expe_cut::CUT_COUNT << " x " << bf16_expe_cut::CUT_DEV_BITS << ")\n"
              << "  candidate ROM   : " << std::setw(6)
              << bf16_expe_cut::CAND_COUNT * bf16_expe_cut::CAND_DEV_BITS << " bits ("
              << bf16_expe_cut::CAND_COUNT << " x " << bf16_expe_cut::CAND_DEV_BITS << ")\n"
              << "  subnormal tail  : " << std::setw(6)
              << bf16_expe_cut::TAIL_COUNT * 7 << " bits ("
              << bf16_expe_cut::TAIL_COUNT << " x 7)\n\n"
              << "Reduction vs LUT   : "
              << std::fixed << std::setprecision(1)
              << (100.0 * (full_lut_bits - cut_bits) / full_lut_bits) << "%\n"
              << "Reduction vs hybrid: "
              << (100.0 * (hybrid_bits - cut_bits) / hybrid_bits) << "%\n\n"
              << (mismatches == 0 ? "PASS\n" : "FAIL\n");

    return mismatches == 0 ? 0 : 1;
}

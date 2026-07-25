// Exhaustive equivalence test: hybrid compressed model vs full LUT model.

#include "bf16_expe_hybrid.hpp"
#include "bf16_expe_lut.hpp"
#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    uint64_t mismatches = 0;

    for (uint32_t raw = 0; raw <= 0xFFFF; ++raw) {
        const uint16_t input = static_cast<uint16_t>(raw);
        const uint16_t expected = bf16_expe_lut_approx(input);
        const uint16_t actual = bf16_expe_hybrid_approx(input);

        if (actual != expected) {
            ++mismatches;
            if (mismatches <= 20) {
                std::cout << "Mismatch input=0x" << std::hex << std::uppercase
                          << std::setw(4) << std::setfill('0') << input
                          << " expected=0x" << std::setw(4) << expected
                          << " actual=0x" << std::setw(4) << actual << std::dec << '\n';
            }
        }
    }

    constexpr int full_lut_bits = 2176 * 16;
    constexpr int dense_rom_bits = 1024 * 16;
    constexpr int sparse_data_bits = 98 * 7 + 8 * 16;
    constexpr int hybrid_bits = dense_rom_bits + sparse_data_bits;

    std::cout << "Full LUT payload: " << full_lut_bits << " bits\n"
              << "Hybrid payload:   " << hybrid_bits << " bits\n"
              << "Reduction:        "
              << (100.0 * (full_lut_bits - hybrid_bits) / full_lut_bits) << "%\n"
              << "Mismatches:       " << mismatches << '\n'
              << (mismatches == 0 ? "PASS\n" : "FAIL\n");

    return mismatches == 0 ? 0 : 1;
}

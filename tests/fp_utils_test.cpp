#include <iostream>
#include <iomanip>
#include <vector>
#include "fp_utils.hpp"

void print_binary16(uint32_t n) {
    // Helper visualization for BF16 (16 bits)
    std::cout << "S EEEEEEEE MMMMMMM\n"; 
    for (int i = 15; i >= 0; i--) {
        std::cout << ((n >> i) & 1);
        if (i == 15 || i == 7) std::cout << " "; 
    }
    std::cout << "\n";
}

int main() {
    std::cout << "--- Universal Hardware FP Utils Test (Configured for BF16) ---\n" << std::endl;

    // Test vectors: Normal, Zero, Inf, NaN, Denormal
    std::vector<uint32_t> test_values = { 
        0x3F80, // 1.0
        0xC000, // -2.0
        0x0000, // +0.0
        0x8000, // -0.0
        0x7F80, // +Inf
        0xFF80, // -Inf
        0x7FC0, // NaN (Standard QNaN)
        0x0001  // Smallest Denormal
    }; 

    bool all_passed = true;

    for (uint32_t original_raw : test_values) {
        // Use BF16 explicitly
        FPType current_type = FPType::BF16;

        std::cout << "Testing Input: 0x" << std::hex << std::setw(4) << std::setfill('0') << original_raw << std::dec << "\n";
        print_binary16(original_raw);

        // 1. Decompose
        FPRaw components = fp_decompose(original_raw, current_type);

        std::cout << "  -> Decomposed: "
                  << "S:" << components.sign 
                  << " Exp:" << components.exponent 
                  << " Mant:0x" << std::hex << components.mantissa << std::dec 
                  << " Hidden:" << components.hidden_bit << "\n";
        
        std::cout << "  -> Status: " 
                  << (components.status.is_zero ? "ZERO " : "")
                  << (components.status.is_inf ? "INF " : "")
                  << (components.status.is_nan ? "NaN " : "")
                  << (components.status.is_denormal ? "DENORM " : "")
                  << (components.hidden_bit ? "NORMAL" : "") // Just for display logic
                  << "\n";

        // 2. Recompose
        uint32_t reconstructed_raw = fp_recompose(components, current_type);

        std::cout << "  -> Reconstructed: 0x" << std::hex << std::setw(4) << std::setfill('0') << reconstructed_raw << std::dec << "\n";

        // 3. Verify
        if ((original_raw & 0xFFFF) == (reconstructed_raw & 0xFFFF)) {
            std::cout << "  [PASS]\n";
        } else {
            // Note: NaNs might not bit-match perfectly depending on mantissa payload preservation logic,
            // but here we try to preserve it.
            std::cout << "  [FAIL]\n";
            all_passed = false;
        }
        std::cout << "--------------------------------------------------\n";
    }

    if (all_passed) std::cout << "\nAll tests passed.\n";
    else std::cout << "\nSome tests failed.\n";

    return 0;
}
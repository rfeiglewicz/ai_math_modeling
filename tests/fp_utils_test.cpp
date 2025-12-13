#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
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

struct TestCase {
    uint32_t value;
    std::string description;
};

int main() {
    std::cout << "--- Universal Hardware FP Utils Test (Configured for BF16) ---\n" << std::endl;

    // Test vectors covering: Normal, Zero, Inf, NaN, Denormal boundaries
    std::vector<TestCase> test_cases = { 
        // --- Basics ---
        {0x3F80, "1.0 (Normal)"},
        {0xC000, "-2.0 (Normal)"},
        {0x0000, "+0.0"},
        {0x8000, "-0.0"},

        // --- Infinity ---
        {0x7F80, "+Inf"},
        {0xFF80, "-Inf"},

        // --- Normal Numbers (Exp: 1..254) ---
        {0x0080, "+Min Normal"},      // Exp=1, Mant=0
        {0x8080, "-Min Normal"},
        {0x7F7F, "+Max Normal"},      // Exp=254, Mant=0x7F (All 1s)
        {0xFF7F, "-Max Normal"},
        {0x3FC0, "+Middle Normal (1.5)"},
        {0x48C9, "+Big Normal number (411648)"},
        {0xC8C9, "-Big Normal number (-411648)"},

        // --- Denormal Numbers (Exp: 0) ---
        {0x0001, "+Min Denormal"},    // Exp=0, Mant=1
        {0x8001, "-Min Denormal"},
        {0x007F, "+Max Denormal"},    // Exp=0, Mant=0x7F
        {0x807F, "-Max Denormal"},
        {0x0040, "+Middle Denormal"}, // Exp=0, Mant=0x40
        {0x8040, "-Middle Denormal"},

        // --- NaNs (Exp: 255, Mant != 0) ---
        {0x7FC0, "+NaN (Standard QNaN)"}, // Exp=255, Mant=0x40
        {0x7F81, "+NaN (Min Payload)"},   // Exp=255, Mant=1
        {0x7FFF, "+NaN (Max Payload)"},   // Exp=255, Mant=0x7F
        {0xFFC0, "-NaN (Standard QNaN)"},
        {0xFF81, "-NaN (Min Payload)"},
        {0xFFFF, "-NaN (Max Payload)"}
    }; 

    bool all_passed = true;

    for (const auto& test : test_cases) {
        uint32_t original_raw = test.value;
        // Use BF16 explicitly
        FPType current_type = FPType::BF16;

        std::cout << "Testing Input: " << test.description << " (0x" << std::hex << std::setw(4) << std::setfill('0') << original_raw << std::dec << ")\n";
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
                  << (components.hidden_bit ? "NORMAL" : "") 
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
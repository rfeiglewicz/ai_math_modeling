#include <iostream>
#include <iomanip>
#include <vector>
#include "fp_utils.hpp"

int main() {
    std::cout << "--- Universal Hardware FP Utils Exhaustive Test (BF16) ---\n" << std::endl;
    std::cout << "Running Exhaustive Test (0x0000 - 0xFFFF)...\n";

    bool all_passed = true;
    int failures = 0;
    const int MAX_FAILURES_TO_PRINT = 10;

    // Iterate through all possible 16-bit values
    for (uint32_t i = 0; i <= 0xFFFF; ++i) {
        // 1. Decompose
        FPRaw components = fp_decompose(i, FPType::BF16);
        
        // 2. Recompose
        uint32_t reconstructed = fp_recompose(components, FPType::BF16);

        // 3. Verify
        // We mask with 0xFFFF because we are testing 16-bit BF16 values packed in uint32_t
        if ((i & 0xFFFF) != (reconstructed & 0xFFFF)) {
            if (failures < MAX_FAILURES_TO_PRINT) {
                std::cout << "Mismatch: Input 0x" << std::hex << std::setw(4) << std::setfill('0') << i 
                          << " -> Output 0x" << std::setw(4) << reconstructed << std::dec << "\n";
                
                if (failures == 0) {
                     std::cout << "  [Debug First Fail] Decomposed: S=" << components.sign 
                               << " Exp=" << components.exponent 
                               << " Mant=0x" << std::hex << components.mantissa 
                               << " Hidden=" << components.hidden_bit << "\n";
                }
            }
            failures++;
            all_passed = false;
        }
    }

    if (failures == 0) {
        std::cout << "\n[SUCCESS] Exhaustive test passed! Checked all 65536 values.\n";
    } else {
        std::cout << "\n[FAIL] Exhaustive test failed! Total mismatches: " << failures << "\n";
    }

    return all_passed ? 0 : 1;
}
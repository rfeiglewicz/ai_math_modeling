#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
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

// =========================================================
// Test: Decompose/Recompose Round-Trip
// =========================================================
bool test_decompose_recompose() {
    std::cout << "=== TEST: Decompose/Recompose Round-Trip ===\n\n";

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
        {0x0080, "+Min Normal"},
        {0x8080, "-Min Normal"},
        {0x7F7F, "+Max Normal"},
        {0xFF7F, "-Max Normal"},
        {0x3FC0, "+Middle Normal (1.5)"},

        // --- Denormal Numbers (Exp: 0) ---
        {0x0001, "+Min Denormal"},
        {0x8001, "-Min Denormal"},
        {0x007F, "+Max Denormal"},
        {0x807F, "-Max Denormal"},
        {0x0040, "+Middle Denormal"},

        // --- NaNs (Exp: 255, Mant != 0) ---
        {0x7FC0, "+NaN (Standard QNaN)"},
        {0x7F81, "+NaN (Min Payload)"},
        {0x7FFF, "+NaN (Max Payload)"},
        {0xFFC0, "-NaN (Standard QNaN)"},
    }; 

    bool all_passed = true;

    for (const auto& test : test_cases) {
        uint32_t original_raw = test.value;
        FPType current_type = FPType::BF16;

        std::cout << "Testing: " << test.description 
                  << " (0x" << std::hex << std::setw(4) << std::setfill('0') << original_raw << std::dec << ")\n";

        FPRaw components = fp_decompose(original_raw, current_type);
        uint32_t reconstructed_raw = fp_recompose(components, current_type);

        if ((original_raw & 0xFFFF) == (reconstructed_raw & 0xFFFF)) {
            std::cout << "  [PASS]\n";
        } else {
            std::cout << "  [FAIL] Got 0x" << std::hex << reconstructed_raw << std::dec << "\n";
            all_passed = false;
        }
    }

    std::cout << "\n";
    return all_passed;
}

// =========================================================
// Test: BF16 to Double Conversion
// =========================================================
struct ConversionTestCase {
    uint32_t bf16_raw;
    double expected_double;
    std::string description;
};

bool test_fp_to_double() {
    std::cout << "=== TEST: BF16 to Double Conversion ===\n\n";

    std::vector<ConversionTestCase> test_cases = {
        // --- Exact Values ---
        {0x0000, 0.0, "+0.0"},
        {0x8000, -0.0, "-0.0"},
        {0x3F80, 1.0, "1.0"},
        {0xBF80, -1.0, "-1.0"},
        {0x4000, 2.0, "2.0"},
        {0xC000, -2.0, "-2.0"},
        {0x3F00, 0.5, "0.5"},
        {0x3E80, 0.25, "0.25"},
        {0x3FC0, 1.5, "1.5"},
        {0x4040, 3.0, "3.0"},
        {0x4080, 4.0, "4.0"},
        {0x40A0, 5.0, "5.0"},
        {0x4120, 10.0, "10.0"},
        {0x42C8, 100.0, "100.0"},

        // --- Powers of 2 ---
        {0x4380, 256.0, "256.0 (2^8)"},
        {0x4700, 32768.0, "32768.0 (2^15)"},
        {0x3B80, 0.00390625, "0.00390625 (2^-8)"},

        // --- Min/Max Normal ---
        {0x0080, 1.1754943508222875e-38, "+Min Normal"},
        {0x7F7F, 3.3895313892515355e+38, "+Max Normal (approx)"},

        // --- Denormals ---
        // Min denormal: 2^(-126) * (1/128) = 2^(-133)
        {0x0001, 9.183549615799121e-41, "+Min Denormal"},
        // Max denormal: 2^(-126) * (127/128)
        {0x007F, 1.1663108012064884e-38, "+Max Denormal"},

        // --- Infinity ---
        {0x7F80, std::numeric_limits<double>::infinity(), "+Inf"},
        {0xFF80, -std::numeric_limits<double>::infinity(), "-Inf"},
    };

    bool all_passed = true;
    const double TOLERANCE = 1e-6; // Relative tolerance for comparison

    for (const auto& test : test_cases) {
        double result = fp_to_double(test.bf16_raw, FPType::BF16);
        
        std::cout << "Testing: " << test.description 
                  << " (0x" << std::hex << std::setw(4) << std::setfill('0') << test.bf16_raw << std::dec << ")\n";
        std::cout << "  Expected: " << std::scientific << std::setprecision(10) << test.expected_double << "\n";
        std::cout << "  Got:      " << result << "\n";

        bool pass = false;

        // Handle special cases
        if (std::isinf(test.expected_double) && std::isinf(result)) {
            pass = (std::signbit(test.expected_double) == std::signbit(result));
        } else if (test.expected_double == 0.0 && result == 0.0) {
            // Check signed zero
            pass = (std::signbit(test.expected_double) == std::signbit(result));
        } else if (test.expected_double != 0.0) {
            // Relative comparison for non-zero values
            double rel_error = std::abs((result - test.expected_double) / test.expected_double);
            pass = (rel_error < TOLERANCE);
        } else {
            pass = (result == test.expected_double);
        }

        if (pass) {
            std::cout << "  [PASS]\n";
        } else {
            std::cout << "  [FAIL]\n";
            all_passed = false;
        }
    }

    // Test NaN separately (NaN != NaN)
    std::cout << "Testing: NaN (0x7FC0)\n";
    double nan_result = fp_to_double(0x7FC0, FPType::BF16);
    if (std::isnan(nan_result)) {
        std::cout << "  [PASS] (result is NaN)\n";
    } else {
        std::cout << "  [FAIL] Expected NaN, got " << nan_result << "\n";
        all_passed = false;
    }

    std::cout << "\n";
    return all_passed;
}

// =========================================================
// Test: ULP Error Calculation
// =========================================================
struct ULPTestCase {
    double ref;
    double val;
    double expected_ulp;
    std::string description;
};

bool test_calculate_ulp_error() {
    std::cout << "=== TEST: ULP Error Calculation ===\n\n";

    std::vector<ULPTestCase> test_cases = {
        // --- Exact Match ---
        {1.0, 1.0, 0.0, "Exact match at 1.0"},
        {0.0, 0.0, 0.0, "Exact match at 0.0"},
        {100.0, 100.0, 0.0, "Exact match at 100.0"},

        // --- Normal Range Errors ---
        // For BF16 at 1.0: 1 ULP = 2^(0 - 7) = 2^(-7) = 0.0078125
        {1.0, 1.0 + 0.0078125, 1.0, "1 ULP error at 1.0"},
        {1.0, 1.0 + 0.015625, 2.0, "2 ULP error at 1.0"},
        {1.0, 1.0 + 0.00390625, 0.5, "0.5 ULP error at 1.0"},

        // For BF16 at 2.0: 1 ULP = 2^(1 - 7) = 2^(-6) = 0.015625
        {2.0, 2.0 + 0.015625, 1.0, "1 ULP error at 2.0"},
        {2.0, 2.0 + 0.03125, 2.0, "2 ULP error at 2.0"},

        // For BF16 at 0.5: 1 ULP = 2^(-1 - 7) = 2^(-8) = 0.00390625
        {0.5, 0.5 + 0.00390625, 1.0, "1 ULP error at 0.5"},

        // --- Larger Values ---
        // At 256.0: 1 ULP = 2^(8 - 7) = 2
        {256.0, 258.0, 1.0, "1 ULP error at 256.0"},
        {256.0, 260.0, 2.0, "2 ULP error at 256.0"},

        // --- Denormal Range ---
        // In denormal range: 1 ULP = 2^(-126 - 7) = 2^(-133)
        // Any value < min_normal uses the same fixed ULP
        {1e-39, 1e-39 + 9.183549615799121e-41, 1.0, "1 ULP error in denormal range"},
        {0.0, 9.183549615799121e-41, 1.0, "1 ULP from zero"},

        // --- Negative Values ---
        {-1.0, -1.0 - 0.0078125, 1.0, "1 ULP error at -1.0"},
        {-2.0, -2.0 - 0.015625, 1.0, "1 ULP error at -2.0"},
    };

    bool all_passed = true;
    const double TOLERANCE = 0.01; // Allow 1% error in ULP calculation

    for (const auto& test : test_cases) {
        double result = calculate_ulp_error(test.ref, test.val, FPType::BF16);
        
        std::cout << "Testing: " << test.description << "\n";
        std::cout << "  Ref: " << std::scientific << test.ref 
                  << ", Val: " << test.val << "\n";
        std::cout << "  Expected ULP: " << std::fixed << std::setprecision(4) << test.expected_ulp 
                  << ", Got: " << result << "\n";

        double error = std::abs(result - test.expected_ulp);
        bool pass = (error < TOLERANCE) || (test.expected_ulp != 0.0 && error / test.expected_ulp < TOLERANCE);

        if (pass) {
            std::cout << "  [PASS]\n";
        } else {
            std::cout << "  [FAIL]\n";
            all_passed = false;
        }
    }

    // Test Infinity cases
    std::cout << "Testing: Inf reference with matching Inf value\n";
    double inf_result = calculate_ulp_error(std::numeric_limits<double>::infinity(), 
                                             std::numeric_limits<double>::infinity(), 
                                             FPType::BF16);
    if (inf_result == 0.0) {
        std::cout << "  [PASS]\n";
    } else {
        std::cout << "  [FAIL] Expected 0.0, got " << inf_result << "\n";
        all_passed = false;
    }

    std::cout << "Testing: Inf reference with non-Inf value\n";
    inf_result = calculate_ulp_error(std::numeric_limits<double>::infinity(), 1000.0, FPType::BF16);
    if (std::isinf(inf_result)) {
        std::cout << "  [PASS]\n";
    } else {
        std::cout << "  [FAIL] Expected Inf, got " << inf_result << "\n";
        all_passed = false;
    }

    std::cout << "Testing: NaN reference\n";
    double nan_result = calculate_ulp_error(std::nan(""), 1.0, FPType::BF16);
    if (std::isnan(nan_result)) {
        std::cout << "  [PASS]\n";
    } else {
        std::cout << "  [FAIL] Expected NaN, got " << nan_result << "\n";
        all_passed = false;
    }

    std::cout << "\n";
    return all_passed;
}

// =========================================================
// Main
// =========================================================
int main() {
    std::cout << "==========================================================\n";
    std::cout << "    Universal Hardware FP Utils Test Suite (BF16)\n";
    std::cout << "==========================================================\n\n";

    bool all_passed = true;

    all_passed &= test_decompose_recompose();
    all_passed &= test_fp_to_double();
    all_passed &= test_calculate_ulp_error();

    std::cout << "==========================================================\n";
    if (all_passed) {
        std::cout << "    ALL TESTS PASSED\n";
    } else {
        std::cout << "    SOME TESTS FAILED\n";
    }
    std::cout << "==========================================================\n";

    return all_passed ? 0 : 1;
}
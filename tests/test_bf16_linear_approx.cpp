#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdint>

// Include the utilities and the generated coefficients
// Adjust paths if your directory structure is different
#include "../src/utils/fp_utils.hpp"
#include "../modeling/coeff_gen/bf16_exp2_coeffs.hpp"

/**
 * @brief Converts a double to BF16 raw bits (uint16_t).
 * 
 * Standard BFloat16 conversion usually involves taking the upper 16 bits 
 * of the Float32 representation.
 */
uint16_t double_to_bf16(double val) {
    // 1. Convert double to float (standard IEEE 754 single precision)
    float f = static_cast<float>(val);
    
    // 2. Bit-cast float to uint32_t to access bits
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    // Handle NaN explicitly
    if (std::isnan(f)) {
        return 0x7FC0; // Canonical qNaN
    }
    
    // 3. Round to Nearest Even (RNE)
    // We add a bias based on the LSB of the target format to handle the tie-breaking rule.
    // If LSB is 0, we add 0x7FFF (round half down/truncate if exactly half).
    // If LSB is 1, we add 0x8000 (round half up to next even).
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    
    // 4. Shift right by 16
    return static_cast<uint16_t>(bits >> 16);
}

int main() {
    // Use the constants defined in the generated header file
    // Range: [-0.25, -0.001953125)
    const float start_range = bf16_exp2::INTERVAL_START;
    const float end_range = bf16_exp2::INTERVAL_END;

    // Iterate through all possible 16-bit patterns
    for (uint32_t i = 0; i <= 0xFFFF; ++i) {
        // Convert raw bits to double for analysis
        double x = fp_to_double(i, FPType::BF16);

        // Filter: Only process values within the specific range
        // We use the float constants from the header for comparison
        if (x >= start_range && x <= end_range) {
            
            // 1. Get the LUT index for this x
            int idx = bf16_exp2::get_lut_index(static_cast<float>(x));

            // 2. Retrieve coefficients (promote to double for calculation)
            double a = static_cast<double>(bf16_exp2::coeffs_a[idx]);
            double b = static_cast<double>(bf16_exp2::coeffs_b[idx]);

            // 3. Calculate Linear Approximation: y = ax + b
            double y = a * x + b;

            // 4. Convert result back to BF16 raw bits
            uint16_t y_raw = double_to_bf16(y);

            // 5. Output in format: HEX_IN HEX_OUT
            std::cout << std::hex << std::uppercase 
                      << std::setw(4) << std::setfill('0') << i 
                      << " " 
                      << std::setw(4) << std::setfill('0') << y_raw 
                      << "\n";
        }
    }

    return 0;
}
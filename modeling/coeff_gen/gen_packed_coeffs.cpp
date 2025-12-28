#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"
#include "ac_int.h"

// Define the fixed-point format parameters
// These match the configuration in src/approximations/bf16_exp2_core.hpp
constexpr int COEFF_I = 1;
constexpr int COEFF_F = 20;
constexpr int COEFF_W = COEFF_I + COEFF_F; 

// Log2E Constant Format 
constexpr int LOG2E_I = 1;
constexpr int LOG2E_F = 22;
constexpr int LOG2E_W = LOG2E_I + LOG2E_F;

// Packed width: 2 coefficients of COEFF_W bits each
constexpr int PACKED_W = 2 * COEFF_W;

int main() {
    std::string output_filename = "modeling/coeff_gen/bf16_exp2_packed_coeffs.hpp";
    std::ofstream out(output_filename);

    if (!out.is_open()) {
        std::cerr << "Error opening file: " << output_filename << std::endl;
        return 1;
    }

    // Header guard and includes
    out << "#ifndef BF16_EXP2_PACKED_COEFFS_HPP\n";
    out << "#define BF16_EXP2_PACKED_COEFFS_HPP\n\n";
    out << "#include \"ac_int.h\"\n";
    out << "#include \"ac_fixed.h\"\n\n";
    out << "namespace bf16_exp2_packed {\n\n";

    out << "constexpr int LUT_SIZE = " << bf16_exp2::LUT_SIZE << ";\n";
    out << "constexpr int COEFF_I = " << COEFF_I << ";\n";
    out << "constexpr int COEFF_F = " << COEFF_F << ";\n";
    out << "constexpr int COEFF_W = " << COEFF_W << ";\n";
    out << "constexpr int PACKED_W = " << PACKED_W << ";\n\n";

    out << "constexpr int LOG2E_I = " << LOG2E_I << ";\n";
    out << "constexpr int LOG2E_F = " << LOG2E_F << ";\n";
    out << "constexpr int LOG2E_W = " << LOG2E_W << ";\n\n";

    // Generate Log2E constant
    typedef ac_fixed<LOG2E_W, LOG2E_I, false> log2e_t;
    log2e_t log2e_val = M_LOG2E; // 1.442695...
    
    // We need to output the raw bits of the fixed point value to ensure exact reconstruction
    // ac_fixed stores bits as an integer.
    ac_int<LOG2E_W, false> log2e_bits = log2e_val.template slc<LOG2E_W>(0);

    out << "// Log2(e) in 1.25 format\n";
    out << "// Value: " << log2e_val.to_double() << "\n";
    // Output as ac_int to ensure bit-exactness in HLS
    out << "static const ac_int<LOG2E_W, false> log2e_int_val = 0x" << std::hex << log2e_bits.to_int64() << ";\n\n";
    
    out << "// Packed coefficients: [ b (26 bits) | a (26 bits) ]\n";
    out << "// Format: unsigned " << COEFF_I << "." << COEFF_F << "\n";
    out << "static const ac_int<PACKED_W, false> coeffs[LUT_SIZE] = {\n";

    typedef ac_fixed<COEFF_W, COEFF_I, false> coeff_t;
    typedef ac_int<PACKED_W, false> packed_t;

    for (int i = 0; i < bf16_exp2::LUT_SIZE; ++i) {
        // Convert float to fixed-point
        coeff_t a_fixed = bf16_exp2::coeffs_a[i];
        coeff_t b_fixed = bf16_exp2::coeffs_b[i];

        // Pack: b is in upper bits, a is in lower bits
        // packed = (b << COEFF_W) | a
        packed_t packed_val = 0;
        packed_t a_bits = a_fixed.template slc<COEFF_W>(0);
        packed_t b_bits = b_fixed.template slc<COEFF_W>(0);

        packed_val = (b_bits << COEFF_W) | a_bits;

        // Output as hex integer literal
        out << "    0x" << std::hex << packed_val.to_int64() << "ULL";
        
        if (i < bf16_exp2::LUT_SIZE - 1) {
            out << ",";
        }
        out << " // Index " << std::dec << i << "\n";
    }

    out << "};\n\n";
    out << "} // namespace bf16_exp2_packed\n\n";
    out << "#endif // BF16_EXP2_PACKED_COEFFS_HPP\n";

    out.close();
    std::cout << "Generated " << output_filename << std::endl;

    return 0;
}

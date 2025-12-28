#include <iostream>
#include <fstream>
#include <iomanip>
#include "bf16_exp2_coeffs.hpp"
#include "ac_fixed.h"
#include "ac_int.h"

// Define the fixed-point format parameters
// These match the configuration in src/approximations/bf16_exp2_core.hpp
constexpr int COEFF_I = 1;
constexpr int COEFF_F = 25;
constexpr int COEFF_W = COEFF_I + COEFF_F; // 26 bits

// Packed width: 2 coefficients * 26 bits = 52 bits
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
    out << "#include \"ac_int.h\"\n\n";
    out << "namespace bf16_exp2_packed {\n\n";

    out << "constexpr int LUT_SIZE = " << bf16_exp2::LUT_SIZE << ";\n";
    out << "constexpr int COEFF_I = " << COEFF_I << ";\n";
    out << "constexpr int COEFF_F = " << COEFF_F << ";\n";
    out << "constexpr int COEFF_W = " << COEFF_W << ";\n";
    out << "constexpr int PACKED_W = " << PACKED_W << ";\n\n";

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

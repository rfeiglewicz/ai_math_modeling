#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include "../src/approximations/bf16_expe_lut.hpp"

int main() {
    const std::string filename_expe_lut = "modeling/golden_ref/bf16_expe_lut_approx_out.txt";

    std::ofstream outfile(filename_expe_lut);

    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename_expe_lut << " for writing.\n";
        return 1;
    }

    std::cout << "Generating tabulated (full-LUT) expe approximation data for all BF16 values...\n";

    // Iterate through all possible negative 16-bit values (matches gen_bf16_exp2_approx.cpp)
    for (uint32_t i = 0x8000; i <= 0xFFFF; ++i) {
        uint16_t input_raw = static_cast<uint16_t>(i);

        // Run the full-LUT expe approximation
        uint16_t output_expe = bf16_expe_lut_approx(input_raw);

        // Write to file: HEX_IN HEX_OUT
        outfile << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw
                << " "
                << std::setw(4) << std::setfill('0') << output_expe << "\n";
    }

    outfile.close();
    std::cout << "Done. Data written to " << filename_expe_lut << "\n";

    return 0;
}

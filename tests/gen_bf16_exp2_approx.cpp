#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include "../src/approximations/bf16_exp2.hpp"

int main() {
    const std::string filename = "modeling/golden_ref/bf16_exp2_approx_out.txt";
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return 1;
    }

    std::cout << "Generating approximation data for all BF16 values...\n";

    // Iterate through all 65536 possible 16-bit values
    for (uint32_t i = 0; i <= 0xFFFF; ++i) {
        uint16_t input_raw = static_cast<uint16_t>(i);
        
        // Run our custom approximation
        uint16_t output_raw = bf16_exp2_approx(input_raw);

        // Write to file: HEX_IN HEX_OUT (no 0x prefix, uppercase)
        outfile << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw 
                << " " 
                << std::setw(4) << std::setfill('0') << output_raw << "\n";
    }

    outfile.close();
    std::cout << "Done. Data written to " << filename << "\n";

    return 0;
}
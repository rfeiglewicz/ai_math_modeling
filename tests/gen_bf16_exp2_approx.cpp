#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include "../src/approximations/bf16_exp2.hpp"

int main() {
    const std::string filename_exp2 = "modeling/golden_ref/bf16_exp2_approx_out.txt";
    const std::string filename_expe = "modeling/golden_ref/bf16_expe_approx_out.txt";
    
    std::ofstream outfile_exp2(filename_exp2);
    std::ofstream outfile_expe(filename_expe);

    if (!outfile_exp2.is_open()) {
        std::cerr << "Error: Could not open file " << filename_exp2 << " for writing.\n";
        return 1;
    }
    if (!outfile_expe.is_open()) {
        std::cerr << "Error: Could not open file " << filename_expe << " for writing.\n";
        return 1;
    }

    std::cout << "Generating approximation data for all BF16 values...\n";

    // Iterate through all possible negative 16-bit values
    for (uint32_t i = 0x8000; i <= 0xFFFF; ++i) {
        uint16_t input_raw = static_cast<uint16_t>(i);
        
        // Run approximation for exp2 (base 2)
        uint16_t output_exp2 = bf16_exp2_approx(input_raw, true);

        // Run approximation for expe (base e)
        uint16_t output_expe = bf16_exp2_approx(input_raw, false);

        // Write to exp2 file
        outfile_exp2 << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw 
                     << " " 
                     << std::setw(4) << std::setfill('0') << output_exp2 << "\n";

        // Write to expe file
        outfile_expe << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw 
                     << " " 
                     << std::setw(4) << std::setfill('0') << output_expe << "\n";
    }

    outfile_exp2.close();
    outfile_expe.close();
    std::cout << "Done. Data written to " << filename_exp2 << " and " << filename_expe << "\n";

    return 0;
}
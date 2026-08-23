#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include "../src/approximations/bf16_exp2.hpp"
#include "../src/approximations/bf16_exp2_optim.hpp"

int main() {
    const std::string filename_exp2 = "modeling/golden_ref/bf16_exp2_approx_out.txt";
    const std::string filename_expe = "modeling/golden_ref/bf16_expe_approx_out.txt";
    // Width-optimised core: same algorithm, narrower datapath. Emitted so the
    // ULP analyzer can rank it next to the other models and confirm it lands
    // on the same BF16 payloads as the production model.
    const std::string filename_exp2_optim = "modeling/golden_ref/bf16_exp2_optim_approx_out.txt";
    const std::string filename_expe_optim = "modeling/golden_ref/bf16_expe_optim_approx_out.txt";

    std::ofstream outfile_exp2(filename_exp2);
    std::ofstream outfile_expe(filename_expe);
    std::ofstream outfile_exp2_optim(filename_exp2_optim);
    std::ofstream outfile_expe_optim(filename_expe_optim);

    if (!outfile_exp2.is_open()) {
        std::cerr << "Error: Could not open file " << filename_exp2 << " for writing.\n";
        return 1;
    }
    if (!outfile_expe.is_open()) {
        std::cerr << "Error: Could not open file " << filename_expe << " for writing.\n";
        return 1;
    }
    if (!outfile_exp2_optim.is_open()) {
        std::cerr << "Error: Could not open file " << filename_exp2_optim << " for writing.\n";
        return 1;
    }
    if (!outfile_expe_optim.is_open()) {
        std::cerr << "Error: Could not open file " << filename_expe_optim << " for writing.\n";
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

        // Same two functions through the width-optimised core
        uint16_t output_exp2_optim = bf16_exp2_optim_approx(input_raw, true);
        uint16_t output_expe_optim = bf16_exp2_optim_approx(input_raw, false);

        // Write to exp2 file
        outfile_exp2 << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw 
                     << " " 
                     << std::setw(4) << std::setfill('0') << output_exp2 << "\n";

        // Write to expe file
        outfile_expe << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw 
                     << " " 
                     << std::setw(4) << std::setfill('0') << output_expe << "\n";

        outfile_exp2_optim << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw
                           << " "
                           << std::setw(4) << std::setfill('0') << output_exp2_optim << "\n";

        outfile_expe_optim << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw
                           << " "
                           << std::setw(4) << std::setfill('0') << output_expe_optim << "\n";
    }

    outfile_exp2.close();
    outfile_expe.close();
    outfile_exp2_optim.close();
    outfile_expe_optim.close();
    std::cout << "Done. Data written to:\n"
              << "  " << filename_exp2 << "\n"
              << "  " << filename_expe << "\n"
              << "  " << filename_exp2_optim << "\n"
              << "  " << filename_expe_optim << "\n";

    return 0;
}
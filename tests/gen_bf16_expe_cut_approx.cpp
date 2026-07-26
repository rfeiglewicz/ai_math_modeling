#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <string>
#include "../src/approximations/bf16_expe_cut.hpp"
#include "../src/approximations/bf16_expe_hybrid.hpp"

int main() {
    const std::string filename_cut = "modeling/golden_ref/bf16_expe_cut_approx_out.txt";
    const std::string filename_hybrid = "modeling/golden_ref/bf16_expe_hybrid_approx_out.txt";

    std::ofstream out_cut(filename_cut);
    std::ofstream out_hybrid(filename_hybrid);

    if (!out_cut.is_open()) {
        std::cerr << "Error: Could not open file " << filename_cut << " for writing.\n";
        return 1;
    }
    if (!out_hybrid.is_open()) {
        std::cerr << "Error: Could not open file " << filename_hybrid << " for writing.\n";
        return 1;
    }

    std::cout << "Generating cut-point ladder and hybrid expe approximation data "
                 "for all negative BF16 values...\n";

    // Iterate through all possible negative 16-bit values (matches gen_bf16_expe_lut_approx.cpp)
    for (uint32_t i = 0x8000; i <= 0xFFFF; ++i) {
        uint16_t input_raw = static_cast<uint16_t>(i);

        uint16_t output_cut = bf16_expe_cut_approx(input_raw);
        uint16_t output_hybrid = bf16_expe_hybrid_approx(input_raw);

        out_cut << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw
                << " " << std::setw(4) << std::setfill('0') << output_cut << "\n";

        out_hybrid << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw
                   << " " << std::setw(4) << std::setfill('0') << output_hybrid << "\n";
    }

    out_cut.close();
    out_hybrid.close();

    std::cout << "Done. Data written to " << filename_cut << "\n"
              << "                  and " << filename_hybrid << "\n";

    return 0;
}

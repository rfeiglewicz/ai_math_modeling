#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <string>
#include "../src/approximations/bf16_expe_poly4.hpp"

int main() {
    const std::string filename = "modeling/golden_ref/bf16_expe_poly4_approx_out.txt";

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return 1;
    }

    std::cout << "Generating degree-4 Horner expe approximation data "
                 "for all negative BF16 values...\n";

    // Iterate through all possible negative 16-bit values
    // (matches gen_bf16_expe_lut_approx.cpp)
    for (uint32_t i = 0x8000; i <= 0xFFFF; ++i) {
        uint16_t input_raw = static_cast<uint16_t>(i);
        uint16_t output = bf16_expe_poly4_approx(input_raw);

        out << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw
            << " " << std::setw(4) << std::setfill('0') << output << "\n";
    }

    out.close();

    std::cout << "Done. Data written to " << filename << "\n";
    return 0;
}

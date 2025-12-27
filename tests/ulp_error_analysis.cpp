#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <cmath>
#include <sstream>
#include "fp_utils.hpp"

void analyze_file(const std::string& input_filename, const std::string& output_filename, bool is_base2) {
    std::ifstream infile(input_filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file " << input_filename << "\n";
        return;
    }

    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file " << output_filename << "\n";
        return;
    }

    std::cout << "Analyzing ULP error for " << (is_base2 ? "exp2" : "expe") << " function...\n";
    std::cout << "Input: " << input_filename << "\n";

    std::string line;
    int line_count = 0;
    int error_count = 0;
    double max_ulp_error = 0.0;
    uint32_t max_ulp_input = 0;
    double total_ulp_error = 0.0;
    int valid_count = 0;

    while (std::getline(infile, line)) {
        // Skip empty lines or comments
        if (line.empty() || line[0] == '/' || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string hex_input, hex_output;
        
        if (!(iss >> hex_input >> hex_output)) {
            std::cerr << "Warning: Could not parse line " << line_count << ": " << line << "\n";
            error_count++;
            continue;
        }

        // Parse hex strings to integers
        uint32_t input_raw = std::stoul(hex_input, nullptr, 16);
        uint32_t output_raw = std::stoul(hex_output, nullptr, 16);

        // Convert BF16 input to double (this is the 'x' value)
        double x_value = fp_to_double(input_raw, FPType::BF16);

        // Convert BF16 output to double (this is the approximation result)
        double approx_result = fp_to_double(output_raw, FPType::BF16);

        // Calculate the ideal reference value using double precision
        double reference;
        if (is_base2) {
            reference = std::exp2(x_value);
        } else {
            reference = std::exp(x_value);
        }

        // Calculate ULP error
        double ulp_error = calculate_ulp_error(reference, approx_result, FPType::BF16);

        // Write to output file: HEX_IN HEX_OUT ULP_ERROR
        outfile << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << input_raw 
                << " " << std::setw(4) << std::setfill('0') << output_raw
                << " " << std::dec << std::fixed << std::setprecision(4);
        
        if (std::isnan(ulp_error)) {
            outfile << "NaN";
        } else if (std::isinf(ulp_error)) {
            outfile << "Inf";
        } else {
            outfile << ulp_error;
            
            // Track statistics (only for finite errors)
            if (ulp_error > max_ulp_error) {
                max_ulp_error = ulp_error;
                max_ulp_input = input_raw;
            }
            total_ulp_error += ulp_error;
            valid_count++;
        }
        outfile << "\n";

        line_count++;
    }

    infile.close();
    outfile.close();

    // Print summary
    std::cout << "=== ULP Error Analysis Summary (" << (is_base2 ? "exp2" : "expe") << ") ===\n";
    std::cout << "Total lines processed: " << line_count << "\n";
    std::cout << "Valid measurements: " << valid_count << "\n";
    std::cout << "Parse errors: " << error_count << "\n";
    
    if (valid_count > 0) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Max ULP error: " << max_ulp_error 
                  << " (at input 0x" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << max_ulp_input << ")\n";
        std::cout << std::dec << "Average ULP error: " << (total_ulp_error / valid_count) << "\n";
    }

    std::cout << "Results written to: " << output_filename << "\n";
    std::cout << "----------------------------------------\n\n";
}

int main() {
    // Analyze exp2 (Base 2)
    analyze_file("modeling/golden_ref/bf16_exp2_approx_out.txt", "modeling/golden_ref/bf16_exp2_ulp.txt", true);

    // Analyze expe (Base e)
    analyze_file("modeling/golden_ref/bf16_expe_approx_out.txt", "modeling/golden_ref/bf16_expe_ulp.txt", false);

    return 0;
}
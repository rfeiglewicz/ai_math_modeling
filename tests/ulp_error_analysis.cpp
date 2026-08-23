#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <map>
#include <vector>
#include "fp_utils.hpp"

// Summary of one analyzed model, used for the cross-model comparison below.
struct AnalysisResult {
    std::string label;
    bool parsed = false;
    int valid_count = 0;
    double max_ulp_error = 0.0;
    uint32_t max_ulp_input = 0;
    double avg_ulp_error = 0.0;
    int over_half_ulp = 0;   // inputs exceeding the 0.5 ULP target
};

void analyze_file(const std::string& input_filename, const std::string& output_filename, bool is_base2,
                  AnalysisResult* summary = nullptr, const std::string& label = "") {
    if (summary) {
        summary->label = label.empty() ? input_filename : label;
    }

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
    int over_half_ulp = 0;

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
            // 0.5 ULP is the correctly-rounded target; allow for binary noise.
            if (ulp_error > 0.5 + 1e-9) {
                over_half_ulp++;
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
        std::cout << std::dec << std::setfill(' ')
                  << "Average ULP error: " << (total_ulp_error / valid_count) << "\n";
        std::cout << "Inputs above 0.5 ULP: " << over_half_ulp
                  << (over_half_ulp == 0 ? "  (correctly rounded)\n" : "\n");
    }

    std::cout << "Results written to: " << output_filename << "\n";
    std::cout << "----------------------------------------\n\n";

    if (summary) {
        summary->parsed = true;
        summary->valid_count = valid_count;
        summary->max_ulp_error = max_ulp_error;
        summary->max_ulp_input = max_ulp_input;
        summary->avg_ulp_error = (valid_count > 0) ? (total_ulp_error / valid_count) : 0.0;
        summary->over_half_ulp = over_half_ulp;
    }
}

// -----------------------------------------------------------------------------
// Cross-model bit-exactness check.
//
// Reads two "HEX_IN HEX_OUT" files and reports how many inputs produce different
// BF16 payloads. The full LUT stores RNE-rounded exp(x), so agreeing with it is
// equivalent to being correctly rounded.
// -----------------------------------------------------------------------------
bool load_outputs(const std::string& filename, std::map<uint32_t, uint32_t>& out) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return false;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '/' || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::string hex_input, hex_output;
        if (!(iss >> hex_input >> hex_output)) {
            continue;
        }
        out[std::stoul(hex_input, nullptr, 16)] = std::stoul(hex_output, nullptr, 16);
    }
    return true;
}

void compare_models(const std::string& reference_name, const std::string& reference_file,
                    const std::string& candidate_name, const std::string& candidate_file) {
    std::map<uint32_t, uint32_t> reference, candidate;
    if (!load_outputs(reference_file, reference) || !load_outputs(candidate_file, candidate)) {
        return;
    }

    int compared = 0;
    int mismatches = 0;
    int shown = 0;

    for (const auto& entry : reference) {
        auto it = candidate.find(entry.first);
        if (it == candidate.end()) {
            continue;
        }
        compared++;
        if (it->second != entry.second) {
            mismatches++;
            if (shown < 10) {
                std::cout << "  Mismatch input=0x" << std::hex << std::uppercase
                          << std::setw(4) << std::setfill('0') << entry.first
                          << "  " << reference_name << "=0x" << std::setw(4) << entry.second
                          << "  " << candidate_name << "=0x" << std::setw(4) << it->second
                          << std::dec << std::setfill(' ') << "\n";
                shown++;
            }
        }
    }

    std::cout << "  " << std::left << std::setw(22) << (candidate_name + " vs " + reference_name)
              << std::right << " compared " << std::setw(6) << compared
              << "  mismatches " << std::setw(6) << mismatches
              << (mismatches == 0 ? "   BIT-EXACT\n" : "   DIFFERS\n");
}

int main() {
    std::vector<AnalysisResult> results(8);

    // Analyze exp2 (Base 2)
    analyze_file("modeling/golden_ref/bf16_exp2_approx_out.txt",
                 "modeling/golden_ref/bf16_exp2_ulp.txt", true,
                 &results[0], "exp2 linear approx");

    // Analyze expe (Base e)
    analyze_file("modeling/golden_ref/bf16_expe_approx_out.txt",
                 "modeling/golden_ref/bf16_expe_ulp.txt", false,
                 &results[1], "expe linear approx");

    // Analyze the width-optimised core. Same algorithm as the linear approx,
    // narrower datapath (see src/approximations/bf16_exp2_optim.hpp), so the
    // rows below should match the two above exactly.
    analyze_file("modeling/golden_ref/bf16_exp2_optim_approx_out.txt",
                 "modeling/golden_ref/bf16_exp2_optim_ulp.txt", true,
                 &results[2], "exp2 optim");

    analyze_file("modeling/golden_ref/bf16_expe_optim_approx_out.txt",
                 "modeling/golden_ref/bf16_expe_optim_ulp.txt", false,
                 &results[3], "expe optim");

    // Analyze expe FULL-LUT model (Base e). Generated by gen_bf16_expe_lut_approx.
    analyze_file("modeling/golden_ref/bf16_expe_lut_approx_out.txt",
                 "modeling/golden_ref/bf16_expe_lut_ulp.txt", false,
                 &results[4], "expe full LUT");

    // Analyze expe HYBRID model (sparse thresholds + dense ROM).
    analyze_file("modeling/golden_ref/bf16_expe_hybrid_approx_out.txt",
                 "modeling/golden_ref/bf16_expe_hybrid_ulp.txt", false,
                 &results[5], "expe hybrid");

    // Analyze expe CUT-POINT LADDER model (shared 2^-f ladder + candidate ROM).
    analyze_file("modeling/golden_ref/bf16_expe_cut_approx_out.txt",
                 "modeling/golden_ref/bf16_expe_cut_ulp.txt", false,
                 &results[6], "expe cut ladder");

    // Analyze expe DEGREE-4 HORNER model (DSP-heavy, storage-light).
    analyze_file("modeling/golden_ref/bf16_expe_poly4_approx_out.txt",
                 "modeling/golden_ref/bf16_expe_poly4_ulp.txt", false,
                 &results[7], "expe degree-4 Horner");

    // ---------------------------------------------------------------------
    // Side-by-side ULP summary
    // ---------------------------------------------------------------------
    std::cout << "========================================================================\n"
              << "ULP comparison across models\n"
              << "========================================================================\n";
    std::cout << std::left << std::setw(22) << "model"
              << std::right << std::setw(9) << "samples"
              << std::setw(11) << "max ULP"
              << std::setw(13) << "avg ULP"
              << std::setw(13) << "> 0.5 ULP" << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (const AnalysisResult& r : results) {
        if (!r.parsed) {
            continue;
        }
        std::cout << std::left << std::setw(22) << r.label
                  << std::right << std::setw(9) << r.valid_count
                  << std::fixed << std::setprecision(4)
                  << std::setw(11) << r.max_ulp_error
                  << std::setw(13) << r.avg_ulp_error
                  << std::setw(13) << r.over_half_ulp << "\n";
    }

    // ---------------------------------------------------------------------
    // Bit-exactness against the correctly-rounded full LUT
    // ---------------------------------------------------------------------
    std::cout << "\n========================================================================\n"
              << "Bit-exactness vs the correctly-rounded full LUT\n"
              << "========================================================================\n";

    const std::string lut_file = "modeling/golden_ref/bf16_expe_lut_approx_out.txt";
    compare_models("full LUT", lut_file,
                   "hybrid", "modeling/golden_ref/bf16_expe_hybrid_approx_out.txt");
    compare_models("full LUT", lut_file,
                   "cut ladder", "modeling/golden_ref/bf16_expe_cut_approx_out.txt");
    compare_models("full LUT", lut_file,
                   "degree-4 Horner",
                   "modeling/golden_ref/bf16_expe_poly4_approx_out.txt");
    compare_models("hybrid", "modeling/golden_ref/bf16_expe_hybrid_approx_out.txt",
                   "cut ladder", "modeling/golden_ref/bf16_expe_cut_approx_out.txt");
    compare_models("cut ladder", "modeling/golden_ref/bf16_expe_cut_approx_out.txt",
                   "degree-4 Horner",
                   "modeling/golden_ref/bf16_expe_poly4_approx_out.txt");
    compare_models("full LUT", lut_file,
                   "linear approx", "modeling/golden_ref/bf16_expe_approx_out.txt");

    // ---------------------------------------------------------------------
    // The width-optimised core is meant to be a drop-in replacement, not a
    // different approximation, so it is checked against the production model
    // rather than against the full LUT. Anything other than BIT-EXACT here
    // means a width in bf16_optim_cfg is too narrow.
    // ---------------------------------------------------------------------
    std::cout << "\n========================================================================\n"
              << "Width-optimised core vs the production linear approximation\n"
              << "========================================================================\n";

    compare_models("exp2 approx", "modeling/golden_ref/bf16_exp2_approx_out.txt",
                   "exp2 optim", "modeling/golden_ref/bf16_exp2_optim_approx_out.txt");
    compare_models("expe approx", "modeling/golden_ref/bf16_expe_approx_out.txt",
                   "expe optim", "modeling/golden_ref/bf16_expe_optim_approx_out.txt");
    compare_models("full LUT", lut_file,
                   "expe optim", "modeling/golden_ref/bf16_expe_optim_approx_out.txt");

    std::cout << "\n";
    return 0;
}
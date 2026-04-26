// =============================================================================
// rtl_compliance_test.cpp
// Verilator testbench - exhaustive verification of all 65536 BF16 values.
// Compares RTL output of bf16_exp2 against the C++ model (bf16_exp2.hpp).
//
// Tests both modes:
//   base2 = 1  ->  2^x approximation
//   base2 = 0  ->  e^x approximation (multiplied by log2(e) internally)
//
// Supports combinational (REGISTER_STAGES=0, default RTL_LATENCY=0) and
// pipelined (REGISTER_STAGES=1) modes.
// Usage: ./obj_dir/Vbf16_exp2 [--latency N]
// =============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include "verilated.h"
#include "Vbf16_exp2.h"

// Include C++ model headers
#include "bf16_exp2.hpp"

// Pipeline latency: 0 for combinational, N for registered (1 reg/stage × N stages)
static int RTL_LATENCY = 0;

struct TestEntry {
    uint16_t input;
    bool     base2;
    uint16_t expected;
};

static inline bool is_nan_bf16(uint16_t v) {
    return ((v >> 7) & 0xFF) == 0xFF && (v & 0x7F) != 0;
}

// ---------------------------------------------------------------------------
// run_test: drives all 65536 BF16 values with a fixed base2 setting,
// resets the DUT first, then returns the number of mismatches found.
// ---------------------------------------------------------------------------
static uint64_t run_test(Vbf16_exp2* dut, bool base2_mode,
                         uint64_t& out_total, uint64_t& out_nan_skipped) {
    // Reset DUT
    dut->clk   = 0;
    dut->rst_n = 0;
    for (int i = 0; i < 4; i++) {
        dut->clk = !dut->clk;
        dut->eval();
    }
    dut->rst_n = 1;
    dut->eval();

    std::vector<TestEntry> pipe;
    pipe.reserve(RTL_LATENCY + 2);

    uint64_t errors     = 0;
    uint64_t total      = 0;
    uint64_t nan_skipped= 0;

    const char* mode_str = base2_mode ? "base2 (2^x)" : "base_e (e^x)";
    std::cout << "\n--- Testing mode: " << mode_str << " ---" << std::endl;

    auto tick_and_capture = [&](uint16_t rtl_out) {
        if ((int)pipe.size() > RTL_LATENCY) {
            TestEntry& te = pipe.front();
            uint16_t got = rtl_out;
            total++;

            // Both NaN -> pass (NaN payload differences are acceptable)
            if (is_nan_bf16(te.expected) && is_nan_bf16(got)) {
                nan_skipped++;
            } else if (got != te.expected) {
                errors++;
                if (errors <= 20) {
                    std::cout << "  MISMATCH input=0x" << std::hex
                              << std::setw(4) << std::setfill('0') << te.input
                              << " expected=0x" << std::setw(4) << te.expected
                              << " rtl=0x"      << std::setw(4) << got
                              << std::dec << std::endl;
                }
            }
            pipe.erase(pipe.begin());
        }
    };

    // Drive all 65536 BF16 values
    for (uint32_t i = 0; i < 65536; i++) {
        uint16_t raw     = static_cast<uint16_t>(i);
        uint16_t cpp_res = bf16_exp2_approx(raw, base2_mode);

        TestEntry te;
        te.input    = raw;
        te.base2    = base2_mode;
        te.expected = cpp_res;
        pipe.push_back(te);

        dut->bf16_in = raw;
        dut->base2   = base2_mode ? 1 : 0;
        dut->clk     = 1;
        dut->eval();
        dut->clk     = 0;
        dut->eval();

        tick_and_capture(dut->bf16_out);
    }

    // Flush pipeline
    for (int i = 0; i < RTL_LATENCY + 2; i++) {
        dut->bf16_in = 0;
        dut->base2   = base2_mode ? 1 : 0;
        dut->clk     = 1;
        dut->eval();
        dut->clk     = 0;
        dut->eval();
        tick_and_capture(dut->bf16_out);
    }

    out_total       = total;
    out_nan_skipped = nan_skipped;
    return errors;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Parse optional latency argument
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--latency") == 0 && i + 1 < argc) {
            RTL_LATENCY = std::atoi(argv[i + 1]);
        }
    }

    Verilated::commandArgs(argc, argv);
    Vbf16_exp2* dut = new Vbf16_exp2;

    std::cout << "=== RTL Compliance Test: bf16_exp2 ===" << std::endl;
    std::cout << "Pipeline latency = " << RTL_LATENCY << " cycles" << std::endl;

    uint64_t total2 = 0, nan2 = 0, errorsBase2 = 0;
    uint64_t totale = 0, nane = 0, errorsBaseE = 0;

    // Run base2 (2^x) test
    errorsBase2 = run_test(dut, /*base2=*/true,  total2, nan2);
    // Run base_e (e^x) test
    errorsBaseE = run_test(dut, /*base2=*/false, totale, nane);

    delete dut;

    // Summary
    std::cout << "\n=== Summary ===" << std::endl;

    std::cout << "[base2 2^x]  checked=" << total2
              << "  nan_skipped=" << nan2
              << "  errors=" << errorsBase2
              << "  " << (errorsBase2 == 0 ? "PASS" : "FAIL") << std::endl;

    std::cout << "[base_e e^x] checked=" << totale
              << "  nan_skipped=" << nane
              << "  errors=" << errorsBaseE
              << "  " << (errorsBaseE == 0 ? "PASS" : "FAIL") << std::endl;

    uint64_t total_errors = errorsBase2 + errorsBaseE;
    std::cout << "\nOVERALL: " << (total_errors == 0 ? "SUCCESS" : "FAILED")
              << " (" << total_errors << " total mismatches)" << std::endl;

    return (total_errors == 0) ? 0 : 1;
}

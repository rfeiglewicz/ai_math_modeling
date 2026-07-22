// =============================================================================
// rtl_expe_lut_compliance_test.cpp
// Verilator testbench -- exhaustive verification of all 65536 BF16 values.
// Compares RTL output of bf16_expe_lut (AXI-Stream) against the C++ model
// bf16_expe_lut_approx() for the full-lookup-table exp(x) design.
//
// Supports combinational (REGISTER_STAGES=0) and pipelined (REGISTER_STAGES=1).
// Usage: ./obj_dir/Vbf16_expe_lut [--latency N]
//
// AXI-Stream protocol:
//   s_axis_tdata  = BF16 input,  s_axis_tvalid/tready
//   m_axis_tdata  = BF16 output, m_axis_tvalid, m_axis_tready
// =============================================================================

#include <iostream>
#include <iomanip>
#include <deque>
#include <cstdint>
#include <cstring>
#include "verilated.h"
#include "Vbf16_expe_lut.h"

// C++ model
#include "bf16_expe_lut.hpp"

// Pipeline latency: 0 for combinational, 4 for pipelined (4 FF stages)
static int RTL_LATENCY = 0;

struct TestEntry {
    uint16_t input;
    uint16_t expected;
};

static inline bool is_nan_bf16(uint16_t v) {
    return ((v >> 7) & 0xFF) == 0xFF && (v & 0x7F) != 0;
}

static void clock_tick(Vbf16_expe_lut* dut) {
    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->eval();
}

static uint64_t run_test(Vbf16_expe_lut* dut,
                         uint64_t& out_total, uint64_t& out_nan_skipped) {
    // Reset DUT
    dut->clk   = 0;
    dut->rst_n = 0;
    dut->s_axis_tvalid = 0;
    dut->m_axis_tready = 1;
    for (int i = 0; i < 4; i++) {
        dut->clk = !dut->clk;
        dut->eval();
    }
    dut->rst_n = 1;
    dut->eval();

    std::deque<TestEntry> expected_q;

    uint64_t errors      = 0;
    uint64_t total       = 0;
    uint64_t nan_skipped = 0;

    std::cout << "\n--- Testing bf16_expe_lut (e^x, full LUT) ---" << std::endl;

    auto check_output = [&]() {
        if (dut->m_axis_tvalid && !expected_q.empty()) {
            TestEntry& te = expected_q.front();
            uint16_t got = dut->m_axis_tdata;
            total++;

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
            expected_q.pop_front();
        }
    };

    // Drive all 65536 BF16 values
    for (uint32_t i = 0; i < 65536; i++) {
        uint16_t raw     = static_cast<uint16_t>(i);
        uint16_t cpp_res = bf16_expe_lut_approx(raw);

        TestEntry te;
        te.input    = raw;
        te.expected = cpp_res;
        expected_q.push_back(te);

        dut->s_axis_tdata  = raw;
        dut->s_axis_tvalid = 1;
        dut->m_axis_tready = 1;

        clock_tick(dut);
        check_output();
    }

    // Flush pipeline
    dut->s_axis_tvalid = 0;
    for (int i = 0; i < RTL_LATENCY + 4; i++) {
        clock_tick(dut);
        check_output();
    }

    out_total       = total;
    out_nan_skipped = nan_skipped;
    return errors;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--latency") == 0 && i + 1 < argc) {
            RTL_LATENCY = std::atoi(argv[i + 1]);
        }
    }

    Verilated::commandArgs(argc, argv);
    Vbf16_expe_lut* dut = new Vbf16_expe_lut;

    std::cout << "=== RTL Compliance Test: bf16_expe_lut (AXI-Stream) ===" << std::endl;
    std::cout << "Pipeline latency = " << RTL_LATENCY << " cycles" << std::endl;

    uint64_t total = 0, nan = 0;
    uint64_t errors = run_test(dut, total, nan);

    delete dut;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "[e^x LUT]  checked=" << total
              << "  nan_skipped=" << nan
              << "  errors=" << errors
              << "  " << (errors == 0 ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nOVERALL: " << (errors == 0 ? "SUCCESS" : "FAILED")
              << " (" << errors << " total mismatches)" << std::endl;

    return (errors == 0) ? 0 : 1;
}

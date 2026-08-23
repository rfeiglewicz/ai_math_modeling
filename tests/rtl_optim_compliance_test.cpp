// =============================================================================
// rtl_optim_compliance_test.cpp
// Verilator testbench for bf16_exp2_optim -- the width-optimised core.
//
// Three things are checked, in this order:
//
//   0. The two C++ models agree. bf16_exp2_optim_approx() must return exactly
//      what bf16_exp2_approx() returns for all 65536 patterns in both modes.
//      If this fails the RTL comparison below is meaningless, so it runs first
//      and aborts.
//
//   1. Exhaustive equivalence. All 65536 BF16 values, both modes, no
//      backpressure, compared against the PRODUCTION model bf16_exp2_approx().
//      Comparing against the production model rather than the optimised one is
//      deliberate: the whole claim of this core is that narrowing the datapath
//      did not change a single output bit, and the test should say exactly
//      that.
//
//   2. AXI-Stream under stress. The same sweep again, this time with
//      pseudo-random tready deassertion and tvalid gaps, so the all-stall
//      pipe_en gating and the valid shift register get exercised. Both cores
//      freeze the entire pipeline on backpressure, and a mistake there shows
//      up as data appearing in the wrong order rather than as wrong values.
//
// Usage: ./obj_dir_optim/Vbf16_exp2_optim [--latency N]
//
// AXI-Stream:
//   s_axis_tdata = BF16 input,  s_axis_tuser = base2,  s_axis_tvalid/tready
//   m_axis_tdata = BF16 result, m_axis_tvalid, m_axis_tready
// =============================================================================

#include <iostream>
#include <iomanip>
#include <deque>
#include <cstdint>
#include <cstring>
#include "verilated.h"
#include "Vbf16_exp2_optim.h"

// The core under test ...
#include "bf16_exp2_optim.hpp"
// ... and the model it must not differ from.
#include "bf16_exp2.hpp"

// Only used to size the pipeline flush; correctness is driven by tvalid.
static int RTL_LATENCY = 0;

struct TestEntry {
    uint16_t input;
    bool     base2;
    uint16_t expected;
};

static inline bool is_nan_bf16(uint16_t v) {
    return ((v >> 7) & 0xFF) == 0xFF && (v & 0x7F) != 0;
}

static uint32_t prng_state = 0x13579BDFu;
static uint32_t prng() {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return prng_state;
}

// ---------------------------------------------------------------------------
// Step 0: the two C++ models must be bit-identical.
// ---------------------------------------------------------------------------
static uint64_t check_models_agree() {
    uint64_t mismatches = 0;
    for (int mode = 0; mode < 2; mode++) {
        const bool base2 = (mode == 0);
        for (uint32_t i = 0; i < 65536; i++) {
            const uint16_t raw = static_cast<uint16_t>(i);
            const uint16_t ref = bf16_exp2_approx(raw, base2);
            const uint16_t opt = bf16_exp2_optim_approx(raw, base2);
            if (ref != opt) {
                if (mismatches < 10) {
                    std::cout << "  MODEL MISMATCH in=0x" << std::hex
                              << std::setw(4) << std::setfill('0') << raw
                              << " production=0x" << std::setw(4) << ref
                              << " optim=0x"      << std::setw(4) << opt
                              << std::dec << std::setfill(' ')
                              << (base2 ? "  [base2]" : "  [base_e]") << std::endl;
                }
                mismatches++;
            }
        }
    }
    return mismatches;
}

// ---------------------------------------------------------------------------
// Clock helper: posedge then negedge.
// ---------------------------------------------------------------------------
static void clock_tick(Vbf16_exp2_optim* dut) {
    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->eval();
}

static void reset_dut(Vbf16_exp2_optim* dut) {
    dut->clk           = 0;
    dut->rst_n         = 0;
    dut->s_axis_tvalid = 0;
    dut->s_axis_tdata  = 0;
    dut->s_axis_tuser  = 1;
    dut->m_axis_tready = 1;
    for (int i = 0; i < 6; i++) { dut->clk = !dut->clk; dut->eval(); }
    dut->rst_n = 1;
    dut->eval();
}

// ---------------------------------------------------------------------------
// Test 1: exhaustive sweep, no backpressure.
// ---------------------------------------------------------------------------
static uint64_t run_exhaustive(Vbf16_exp2_optim* dut, bool base2_mode,
                               uint64_t& out_total, uint64_t& out_nan_checked) {
    reset_dut(dut);

    std::deque<TestEntry> expected_q;
    uint64_t errors = 0, total = 0, nan_checked = 0;

    const char* mode_str = base2_mode ? "base2 (2^x)" : "base_e (e^x)";
    std::cout << "\n--- Exhaustive sweep, mode: " << mode_str << " ---" << std::endl;

    auto check_output = [&]() {
        if (dut->m_axis_tvalid && !expected_q.empty()) {
            TestEntry& te = expected_q.front();
            uint16_t got = dut->m_axis_tdata;
            total++;

            if (is_nan_bf16(te.expected)) nan_checked++;

            if (got != te.expected) {
                errors++;
                if (errors <= 20) {
                    std::cout << "  MISMATCH input=0x" << std::hex
                              << std::setw(4) << std::setfill('0') << te.input
                              << " expected=0x" << std::setw(4) << te.expected
                              << " rtl=0x"      << std::setw(4) << got
                              << std::dec << std::setfill(' ') << std::endl;
                }
            }
            expected_q.pop_front();
        }
    };

    for (uint32_t i = 0; i < 65536; i++) {
        uint16_t raw = static_cast<uint16_t>(i);

        TestEntry te;
        te.input    = raw;
        te.base2    = base2_mode;
        // Production model on purpose: see the file header.
        te.expected = bf16_exp2_approx(raw, base2_mode);
        expected_q.push_back(te);

        dut->s_axis_tdata  = raw;
        dut->s_axis_tuser  = base2_mode ? 1 : 0;
        dut->s_axis_tvalid = 1;
        dut->m_axis_tready = 1;

        clock_tick(dut);
        check_output();
    }

    // Flush the pipeline.
    dut->s_axis_tvalid = 0;
    for (int i = 0; i < RTL_LATENCY + 8; i++) {
        clock_tick(dut);
        check_output();
    }

    out_total       = total;
    out_nan_checked = nan_checked;
    return errors;
}

// ---------------------------------------------------------------------------
// Test 2: same sweep with backpressure and input gaps.
//
// Proper AXI-Stream master/slave behaviour:
//   - tdata/tuser/tvalid stay stable until the handshake completes
//   - handshake conditions are sampled with the clock low, before the edge
//   - m_axis_tready toggles pseudo-randomly, so the pipeline stalls often
// ---------------------------------------------------------------------------
static uint64_t run_backpressure(Vbf16_exp2_optim* dut, bool base2_mode,
                                 uint64_t& out_total) {
    reset_dut(dut);
    prng_state = base2_mode ? 0x13579BDFu : 0x2468ACE0u;

    std::deque<TestEntry> expected_q;
    uint64_t errors = 0, total = 0;

    const char* mode_str = base2_mode ? "base2 (2^x)" : "base_e (e^x)";
    std::cout << "\n--- Backpressure sweep, mode: " << mode_str << " ---" << std::endl;

    uint32_t next_idx = 0;
    bool     pending  = false;
    uint16_t pend_raw = 0;
    uint16_t pend_exp = 0;

    const uint32_t COUNT = 65536;
    // Generous bound: worst case is a stall on most cycles.
    const uint64_t MAX_CYCLES = 12ull * COUNT + 4096;
    uint64_t cycles = 0;

    while ((next_idx < COUNT || pending || !expected_q.empty()) && cycles < MAX_CYCLES) {
        cycles++;

        // --- master side: load a new item if the bus is free ---------------
        if (!pending && next_idx < COUNT) {
            // Leave an occasional gap so tvalid really does drop mid-stream.
            if ((prng() & 7) != 0) {
                pend_raw = static_cast<uint16_t>(next_idx);
                pend_exp = bf16_exp2_approx(pend_raw, base2_mode);
                pending  = true;
            }
        }

        if (pending) {
            dut->s_axis_tvalid = 1;
            dut->s_axis_tdata  = pend_raw;
            dut->s_axis_tuser  = base2_mode ? 1 : 0;
        } else {
            dut->s_axis_tvalid = 0;
        }

        // --- slave side: random backpressure -------------------------------
        dut->m_axis_tready = ((prng() & 3) != 0) ? 1 : 0;

        // Settle combinationally, then sample what the edge will capture.
        dut->eval();
        const bool input_accepted  = dut->s_axis_tvalid && dut->s_axis_tready;
        const bool output_consumed = dut->m_axis_tvalid && dut->m_axis_tready;
        const uint16_t out_data    = dut->m_axis_tdata;

        // Push before popping. With REGISTER_STAGES=0 the core is purely
        // combinational, so the accepted input and its result appear in the
        // SAME cycle and the item has to be in the queue before the output is
        // matched against it. With REGISTER_STAGES=1 there is always at least
        // one older item in flight whenever an output is consumed, so pushing
        // to the back first cannot disturb the front. One order works for both.
        if (input_accepted) {
            expected_q.push_back({pend_raw, base2_mode, pend_exp});
            pending = false;
            next_idx++;
        }

        if (output_consumed) {
            if (expected_q.empty()) {
                errors++;
                if (errors <= 20)
                    std::cout << "  SPURIOUS OUTPUT 0x" << std::hex << out_data
                              << std::dec << std::endl;
            } else {
                TestEntry te = expected_q.front();
                expected_q.pop_front();
                total++;
                if (out_data != te.expected) {
                    errors++;
                    if (errors <= 20) {
                        std::cout << "  MISMATCH input=0x" << std::hex
                                  << std::setw(4) << std::setfill('0') << te.input
                                  << " expected=0x" << std::setw(4) << te.expected
                                  << " rtl=0x"      << std::setw(4) << out_data
                                  << std::dec << std::setfill(' ') << std::endl;
                    }
                }
            }
        }

        clock_tick(dut);
    }

    if (cycles >= MAX_CYCLES) {
        std::cout << "  TIMEOUT after " << cycles << " cycles, "
                  << expected_q.size() << " results still in flight" << std::endl;
        errors++;
    }

    out_total = total;
    return errors;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--latency") == 0 && i + 1 < argc)
            RTL_LATENCY = std::atoi(argv[i + 1]);
    }

    Verilated::commandArgs(argc, argv);

    std::cout << "=== RTL Compliance Test: bf16_exp2_optim (AXI-Stream) ===" << std::endl;
    std::cout << "Pipeline latency = " << RTL_LATENCY << " cycles" << std::endl;

    // --- step 0 -----------------------------------------------------------
    std::cout << "\n--- C++ model cross-check: optim vs production ---" << std::endl;
    const uint64_t model_mismatches = check_models_agree();
    std::cout << "  compared 131072 values, mismatches " << model_mismatches
              << "  " << (model_mismatches == 0 ? "BIT-EXACT" : "DIVERGED") << std::endl;
    if (model_mismatches != 0) {
        std::cout << "\nOVERALL: FAILED (the C++ models already disagree; "
                     "fix that before looking at the RTL)" << std::endl;
        return 1;
    }

    Vbf16_exp2_optim* dut = new Vbf16_exp2_optim;

    uint64_t total2 = 0, nan2 = 0, totale = 0, nane = 0;
    uint64_t err_sweep2 = run_exhaustive(dut, /*base2=*/true,  total2, nan2);
    uint64_t err_sweepe = run_exhaustive(dut, /*base2=*/false, totale, nane);

    uint64_t bp_total2 = 0, bp_totale = 0;
    uint64_t err_bp2 = run_backpressure(dut, /*base2=*/true,  bp_total2);
    uint64_t err_bpe = run_backpressure(dut, /*base2=*/false, bp_totale);

    delete dut;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "[sweep base2 2^x]   checked=" << total2
              << "  nan_checked=" << nan2
              << "  errors=" << err_sweep2
              << "  " << (err_sweep2 == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << "[sweep base_e e^x]  checked=" << totale
              << "  nan_checked=" << nane
              << "  errors=" << err_sweepe
              << "  " << (err_sweepe == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << "[stall base2 2^x]   checked=" << bp_total2
              << "  errors=" << err_bp2
              << "  " << (err_bp2 == 0 ? "PASS" : "FAIL") << std::endl;
    std::cout << "[stall base_e e^x]  checked=" << bp_totale
              << "  errors=" << err_bpe
              << "  " << (err_bpe == 0 ? "PASS" : "FAIL") << std::endl;

    const uint64_t total_errors = err_sweep2 + err_sweepe + err_bp2 + err_bpe;
    std::cout << "\nReference for every comparison above: bf16_exp2_approx(), "
                 "the production model." << std::endl;
    std::cout << "OVERALL: " << (total_errors == 0 ? "SUCCESS" : "FAILED")
              << " (" << total_errors << " total mismatches)" << std::endl;

    return (total_errors == 0) ? 0 : 1;
}

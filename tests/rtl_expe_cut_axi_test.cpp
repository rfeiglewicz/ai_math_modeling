// =============================================================================
// rtl_expe_cut_axi_test.cpp
// Verilator testbench -- AXI-Stream protocol & backpressure for bf16_expe_cut.
//
// The compliance test keeps m_axis_tready asserted for the whole run, so it
// never exercises the handshake. This one does:
//   1. BACKPRESSURE   - random m_axis_tready, every result still correct
//   2. LONG STALL     - m_axis_tready=0 for many cycles, output must hold
//   3. INPUT GAPS     - random s_axis_tvalid, no spurious outputs
//   4. COMBINED       - all 65536 BF16 with random valid AND random ready
//
// Build & run: make rtl_expe_cut_axi_test
// =============================================================================

#include "bf16_expe_cut.hpp"
#include "Vbf16_expe_cut.h"
#include "verilated.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>

namespace {

uint32_t prng_state = 0xDEADBEEF;
uint32_t prng() {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return prng_state;
}

struct Expected {
    uint16_t input;
    uint16_t output;
};

struct TickResult {
    bool     input_accepted;
    bool     output_consumed;
    uint16_t output_data;
};

// Captures the handshake state that the posedge will sample, then clocks.
TickResult axi_tick(Vbf16_expe_cut* dut) {
    TickResult r;
    dut->eval();
    r.input_accepted  = dut->s_axis_tvalid && dut->s_axis_tready;
    r.output_consumed = dut->m_axis_tvalid && dut->m_axis_tready;
    r.output_data     = dut->m_axis_tdata;

    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->eval();
    return r;
}

void reset_dut(Vbf16_expe_cut* dut) {
    dut->clk = 0;
    dut->rst_n = 0;
    dut->s_axis_tvalid = 0;
    dut->s_axis_tdata = 0;
    dut->m_axis_tready = 0;
    for (int i = 0; i < 6; ++i) {
        dut->clk = !dut->clk;
        dut->eval();
    }
    dut->rst_n = 1;
    dut->eval();
}

int g_errors = 0;

void report(const char* test, uint16_t input, uint16_t expected, uint16_t got) {
    ++g_errors;
    if (g_errors <= 30) {
        std::cout << "  [" << test << "] MISMATCH in=0x" << std::hex << std::uppercase
                  << std::setw(4) << std::setfill('0') << input
                  << " exp=0x" << std::setw(4) << expected
                  << " got=0x" << std::setw(4) << got
                  << std::dec << std::setfill(' ') << '\n';
    }
}

// ---------------------------------------------------------------------------
// Generic driver: streams `count` inputs while randomising the handshake.
//
// valid_pct / ready_pct are percentages; 100 means "always asserted".
// AXI requires tdata/tvalid to stay stable until the handshake completes, so
// the pending item is only replaced after it has been accepted.
// ---------------------------------------------------------------------------
uint64_t run_stream(Vbf16_expe_cut* dut, const char* test, int count,
                    int valid_pct, int ready_pct, int stall_every, int stall_len) {
    reset_dut(dut);

    const int errors_before = g_errors;
    std::deque<Expected> queue;
    uint64_t checked = 0;
    int next_index = 0;

    bool pending = false;
    uint16_t pend_data = 0;

    int64_t cycle = 0;
    int stall_counter = 0;
    int cycles_without_progress = 0;
    const int max_idle = 10000;

    // Held-output check: while m_axis_tready is low, m_axis_tdata must not move.
    bool holding = false;
    uint16_t held_value = 0;

    while ((next_index < count || pending || !queue.empty())
           && cycles_without_progress < max_idle) {

        // ---- drive slave side ----
        if (!pending && next_index < count) {
            if (valid_pct >= 100 || static_cast<int>(prng() % 100) < valid_pct) {
                pend_data = static_cast<uint16_t>(next_index);
                pending = true;
            }
        }
        dut->s_axis_tvalid = pending ? 1 : 0;
        dut->s_axis_tdata = pending ? pend_data : 0;

        // ---- drive master side ----
        // The stall schedule is cycle based on purpose: a stall also freezes
        // s_axis_tready, so an index-based schedule would re-arm itself forever
        // and deadlock the testbench rather than the DUT.
        bool ready;
        if (stall_counter > 0) {
            ready = false;
            --stall_counter;
        } else if (stall_every > 0 && cycle > 0 && (cycle % stall_every) == 0) {
            stall_counter = stall_len - 1;
            ready = false;
        } else {
            ready = (ready_pct >= 100) || (static_cast<int>(prng() % 100) < ready_pct);
        }
        dut->m_axis_tready = ready ? 1 : 0;
        ++cycle;

        dut->eval();

        // Output must remain stable while it is not being consumed.
        if (dut->m_axis_tvalid && !ready) {
            const uint16_t now = dut->m_axis_tdata;
            if (holding && now != held_value) {
                ++g_errors;
                if (g_errors <= 30) {
                    std::cout << "  [" << test << "] output moved during stall: 0x"
                              << std::hex << held_value << " -> 0x" << now
                              << std::dec << '\n';
                }
            }
            holding = true;
            held_value = now;
        } else {
            holding = false;
        }

        const TickResult r = axi_tick(dut);

        bool progressed = false;
        if (r.input_accepted) {
            queue.push_back({pend_data, bf16_expe_cut_approx(pend_data)});
            pending = false;
            ++next_index;
            progressed = true;
        }
        if (r.output_consumed) {
            if (queue.empty()) {
                ++g_errors;
                std::cout << "  [" << test << "] output with empty queue\n";
            } else {
                const Expected e = queue.front();
                queue.pop_front();
                ++checked;
                if (r.output_data != e.output) {
                    report(test, e.input, e.output, r.output_data);
                }
            }
            progressed = true;
        }
        cycles_without_progress = progressed ? 0 : cycles_without_progress + 1;
    }

    if (cycles_without_progress >= max_idle) {
        ++g_errors;
        std::cout << "  [" << test << "] pipeline stalled permanently\n";
    }
    if (checked != static_cast<uint64_t>(count)) {
        ++g_errors;
        std::cout << "  [" << test << "] expected " << count << " results, got "
                  << checked << '\n';
    }

    std::cout << "  " << std::left << std::setw(16) << test << std::right
              << " results " << std::setw(6) << checked
              << "   errors " << std::setw(3) << (g_errors - errors_before)
              << (g_errors == errors_before ? "   PASS\n" : "   FAIL\n");
    return checked;
}

} // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vbf16_expe_cut;

    std::cout << "AXI-Stream protocol tests for bf16_expe_cut\n"
              << "-------------------------------------------\n";

    run_stream(dut, "BACKPRESSURE", 4096, 100, 50, 0, 0);
    run_stream(dut, "LONG STALL", 2048, 100, 100, 64, 20);
    run_stream(dut, "INPUT GAPS", 4096, 40, 100, 0, 0);
    run_stream(dut, "FULL SWEEP", 65536, 70, 60, 0, 0);

    std::cout << "-------------------------------------------\n"
              << "Total errors : " << g_errors << '\n'
              << (g_errors == 0 ? "PASS\n" : "FAIL\n");

    delete dut;
    return g_errors == 0 ? 0 : 1;
}

// =============================================================================
// rtl_expe_equivalence_test.cpp
//
// Proves that every BF16 exp(x) implementation is a drop-in replacement for
// every other one. All cores are driven from a single AXI-Stream input inside
// bf16_expe_equiv_top and checked against each other cycle by cycle.
//
// What is checked:
//   1. handshake      - s_axis_tready and m_axis_tvalid must be identical on
//                       every cycle for every core
//   2. data           - m_axis_tdata must match bit for bit whenever valid,
//                       INCLUDING NaN inputs (no NaN results are skipped)
//   3. latency        - measured, must be the same for all cores
//   4. coverage       - all 65536 BF16 patterns, plus a backpressure pass
//
// Usage: ./Vbf16_expe_equiv_top
// =============================================================================

#include <verilated.h>
#include "Vbf16_expe_equiv_top.h"

#include <cstdint>
#include <cstdio>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int NUM_CORES = 7;

const char* kCoreName[NUM_CORES] = {
    "exp2 baseline",
    "exp2 optimised",
    "expe full-lut",
    "expe hybrid",
    "expe cut-ladder",
    "expe poly4",
    "expe poly4 dsp",
};

struct CoreView {
    uint16_t data;
    bool     tvalid;
    bool     tready;
};

CoreView view(const Vbf16_expe_equiv_top* dut, int idx) {
    switch (idx) {
        case 0: return {dut->o_exp2_tdata,      !!dut->o_exp2_tvalid,      !!dut->o_exp2_tready};
        case 1: return {dut->o_exp2opt_tdata,   !!dut->o_exp2opt_tvalid,   !!dut->o_exp2opt_tready};
        case 2: return {dut->o_lut_tdata,       !!dut->o_lut_tvalid,       !!dut->o_lut_tready};
        case 3: return {dut->o_hybrid_tdata,    !!dut->o_hybrid_tvalid,    !!dut->o_hybrid_tready};
        case 4: return {dut->o_cut_tdata,       !!dut->o_cut_tvalid,       !!dut->o_cut_tready};
        case 5: return {dut->o_poly4_tdata,     !!dut->o_poly4_tvalid,     !!dut->o_poly4_tready};
        default:return {dut->o_poly4dsp_tdata,  !!dut->o_poly4dsp_tvalid,  !!dut->o_poly4dsp_tready};
    }
}

void tick(Vbf16_expe_equiv_top* dut) {
    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->eval();
}

void reset(Vbf16_expe_equiv_top* dut) {
    dut->clk           = 0;
    dut->rst_n         = 0;
    dut->s_axis_tvalid = 0;
    dut->s_axis_tdata  = 0;
    dut->m_axis_tready = 1;
    for (int i = 0; i < 8; i++) tick(dut);
    dut->rst_n = 1;
    dut->eval();
}

struct Stats {
    uint64_t compared      = 0;   // result sets checked
    uint64_t data_errors   = 0;   // cores disagreeing on data
    uint64_t valid_errors  = 0;   // cores disagreeing on tvalid
    uint64_t ready_errors  = 0;   // cores disagreeing on tready
    uint64_t nan_inputs    = 0;   // NaN inputs seen (compared, never skipped)
};

// -----------------------------------------------------------------------------
// Measure output latency: cycles from accepting one input to its result.
// -----------------------------------------------------------------------------
int measure_latency(Vbf16_expe_equiv_top* dut, int core) {
    reset(dut);
    dut->m_axis_tready = 1;
    dut->s_axis_tdata  = 0xBF80;   // -1.0
    dut->s_axis_tvalid = 1;
    dut->eval();

    tick(dut);                     // this edge accepts the sample
    dut->s_axis_tvalid = 0;
    dut->eval();

    for (int cycles = 1; cycles <= 64; cycles++) {
        if (view(dut, core).tvalid) return cycles;
        tick(dut);
    }
    return -1;
}

// -----------------------------------------------------------------------------
// Sweep every BF16 pattern and cross-check all cores.
// `stall_rate` in percent drives random backpressure and input gaps.
// -----------------------------------------------------------------------------
Stats sweep(Vbf16_expe_equiv_top* dut, int stall_rate, uint32_t seed) {
    reset(dut);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pct(0, 99);

    Stats st;
    std::deque<uint16_t> inflight;      // inputs accepted, awaiting results

    uint32_t next = 0;
    const uint32_t kTotal = 65536;
    int drain = 0;

    while (next < kTotal || !inflight.empty() || drain < 64) {
        const bool feed  = (next < kTotal) && (stall_rate == 0 || pct(rng) >= stall_rate);
        const bool ready = (stall_rate == 0) || (pct(rng) >= stall_rate);

        dut->s_axis_tdata  = feed ? static_cast<uint16_t>(next) : 0;
        dut->s_axis_tvalid = feed;
        dut->m_axis_tready = ready;
        dut->eval();

        // ---- tready must agree across cores ----
        const bool ready0 = view(dut, 0).tready;
        for (int c = 1; c < NUM_CORES; c++) {
            if (view(dut, c).tready != ready0) {
                if (st.ready_errors < 10) {
                    std::cout << "  TREADY MISMATCH: " << kCoreName[0] << "=" << ready0
                              << " vs " << kCoreName[c] << "=" << view(dut, c).tready
                              << std::endl;
                }
                st.ready_errors++;
            }
        }

        // ---- tvalid must agree across cores ----
        const bool valid0 = view(dut, 0).tvalid;
        for (int c = 1; c < NUM_CORES; c++) {
            if (view(dut, c).tvalid != valid0) {
                if (st.valid_errors < 10) {
                    std::cout << "  TVALID MISMATCH: " << kCoreName[0] << "=" << valid0
                              << " vs " << kCoreName[c] << "=" << view(dut, c).tvalid
                              << std::endl;
                }
                st.valid_errors++;
            }
        }

        // ---- data must agree across cores ----
        if (valid0 && ready && !inflight.empty()) {
            const uint16_t in = inflight.front();
            inflight.pop_front();
            st.compared++;

            const bool is_nan = (((in >> 7) & 0xFF) == 0xFF) && ((in & 0x7F) != 0);
            if (is_nan) st.nan_inputs++;

            const uint16_t ref = view(dut, 0).data;
            for (int c = 1; c < NUM_CORES; c++) {
                const uint16_t got = view(dut, c).data;
                if (got != ref) {
                    if (st.data_errors < 20) {
                        std::cout << "  DATA MISMATCH input=0x" << std::hex << std::setw(4)
                                  << std::setfill('0') << in
                                  << "  " << kCoreName[0] << "=0x" << std::setw(4) << ref
                                  << "  " << kCoreName[c] << "=0x" << std::setw(4) << got
                                  << std::dec << std::setfill(' ') << std::endl;
                    }
                    st.data_errors++;
                }
            }
        }

        const bool accepted = feed && ready0;
        if (accepted) {
            inflight.push_back(static_cast<uint16_t>(next));
            next++;
        }

        tick(dut);

        if (next >= kTotal && inflight.empty()) drain++;
    }

    return st;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vbf16_expe_equiv_top;

    std::cout << "=== Cross-implementation equivalence: BF16 exp(x) ===" << std::endl;
    std::cout << "Cores under test: " << NUM_CORES << std::endl << std::endl;

    bool ok = true;

    // -------------------------------------------------------------------------
    // 1. Latency must be identical
    // -------------------------------------------------------------------------
    std::cout << "--- Latency ---" << std::endl;
    int lat0 = -1;
    bool latency_ok = true;
    for (int c = 0; c < NUM_CORES; c++) {
        const int lat = measure_latency(dut, c);
        if (c == 0) lat0 = lat;
        std::cout << "  " << std::left << std::setw(18) << kCoreName[c]
                  << std::right << lat << " cycles"
                  << (lat == lat0 ? "" : "   <-- DIFFERENT") << std::endl;
        if (lat != lat0 || lat < 0) latency_ok = false;
    }
    std::cout << "  " << (latency_ok ? "PASS" : "FAIL")
              << "  (all cores = " << lat0 << " cycles)" << std::endl << std::endl;
    ok = ok && latency_ok;

    // -------------------------------------------------------------------------
    // 2. Full sweep, no backpressure
    // -------------------------------------------------------------------------
    std::cout << "--- Full sweep, no backpressure ---" << std::endl;
    const Stats s1 = sweep(dut, 0, 1);
    std::cout << "  compared=" << s1.compared
              << "  nan_inputs=" << s1.nan_inputs
              << "  data_err=" << s1.data_errors
              << "  valid_err=" << s1.valid_errors
              << "  ready_err=" << s1.ready_errors
              << "  " << ((s1.data_errors || s1.valid_errors || s1.ready_errors) ? "FAIL" : "PASS")
              << std::endl << std::endl;
    ok = ok && !s1.data_errors && !s1.valid_errors && !s1.ready_errors && s1.compared == 65536;

    // -------------------------------------------------------------------------
    // 3. Full sweep with random backpressure and input gaps
    // -------------------------------------------------------------------------
    std::cout << "--- Full sweep, 30% random stalls ---" << std::endl;
    const Stats s2 = sweep(dut, 30, 12345);
    std::cout << "  compared=" << s2.compared
              << "  nan_inputs=" << s2.nan_inputs
              << "  data_err=" << s2.data_errors
              << "  valid_err=" << s2.valid_errors
              << "  ready_err=" << s2.ready_errors
              << "  " << ((s2.data_errors || s2.valid_errors || s2.ready_errors) ? "FAIL" : "PASS")
              << std::endl << std::endl;
    ok = ok && !s2.data_errors && !s2.valid_errors && !s2.ready_errors && s2.compared == 65536;

    // -------------------------------------------------------------------------
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Latency identical      : " << (latency_ok ? "yes" : "NO") << std::endl;
    std::cout << "Data identical         : "
              << ((!s1.data_errors && !s2.data_errors) ? "yes" : "NO") << std::endl;
    std::cout << "Handshake identical    : "
              << ((!s1.valid_errors && !s2.valid_errors &&
                   !s1.ready_errors && !s2.ready_errors) ? "yes" : "NO") << std::endl;
    std::cout << "NaN inputs compared    : " << s1.nan_inputs
              << " (none skipped)" << std::endl;
    std::cout << std::endl << "OVERALL: " << (ok ? "SUCCESS" : "FAILED") << std::endl;

    dut->final();
    delete dut;
    return ok ? 0 : 1;
}

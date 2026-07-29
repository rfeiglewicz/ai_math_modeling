// =============================================================================
// rtl_axi_stream_test.cpp
// Verilator testbench -- AXI-Stream protocol & backpressure verification.
//
// Tests (pipelined mode, REGISTER_STAGES=1):
//   1. BASIC FLOW       - continuous valid+ready, data arrives correctly
//   2. BACKPRESSURE      - random m_axis_tready, data still correct
//   3. STALL + HOLD      - m_axis_tready=0 for many cycles, output holds
//   4. INPUT GAPS        - random s_axis_tvalid, no spurious outputs
//   5. PROTOCOL CHECKS   - s_axis_tready=0 when pipeline is stalled
//   6. COMBINED STRESS   - random valid + random ready simultaneously
//   7. DATA INTEGRITY    - all 65536 BF16 under random backpressure
//
// Build & run: make rtl_axi_test
// =============================================================================

#include <iostream>
#include <iomanip>
#include <deque>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "verilated.h"
#include "Vbf16_exp2.h"
#include "bf16_exp2.hpp"

// ---- Helpers ----------------------------------------------------------------

static inline bool is_nan_bf16(uint16_t v) {
    return ((v >> 7) & 0xFF) == 0xFF && (v & 0x7F) != 0;
}

static uint32_t prng_state = 0xDEADBEEF;
static uint32_t prng() {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return prng_state;
}

static void reset_dut(Vbf16_exp2* dut) {
    dut->clk = 0;
    dut->rst_n = 0;
    dut->s_axis_tvalid = 0;
    dut->s_axis_tdata  = 0;
    dut->s_axis_tuser  = 1;
    dut->m_axis_tready = 0;
    for (int i = 0; i < 6; i++) { dut->clk = !dut->clk; dut->eval(); }
    dut->rst_n = 1;
    dut->eval();
}

// ---- AXI-Stream cycle result ------------------------------------------------
// Captures handshake state BEFORE the clock edge, then advances the clock.
// This is critical: the handshake condition (tvalid && tready) is sampled at
// the rising edge. We must evaluate combinational logic first to get the
// correct tready, then record what will be captured at the edge.

struct TickResult {
    bool     input_accepted;   // s_axis_tvalid && s_axis_tready at posedge
    bool     output_consumed;  // m_axis_tvalid && m_axis_tready at posedge
    uint16_t output_data;      // m_axis_tdata (registered, from previous cycle)
};

static TickResult axi_tick(Vbf16_exp2* dut) {
    TickResult r;

    // Settle combinational paths with current inputs (clk still low)
    dut->eval();

    // Record handshake conditions that will be sampled at posedge
    r.input_accepted  = dut->s_axis_tvalid && dut->s_axis_tready;
    r.output_consumed = dut->m_axis_tvalid && dut->m_axis_tready;
    r.output_data     = dut->m_axis_tdata;

    // Rising edge
    dut->clk = 1;
    dut->eval();
    // Falling edge
    dut->clk = 0;
    dut->eval();

    return r;
}

// ---- Expected result queue --------------------------------------------------

struct Expected {
    uint16_t input;
    uint16_t expected;
};

static int g_test_errors = 0;

static bool compare(uint16_t expected, uint16_t got, uint16_t input, const char* test) {
    // NaN is compared bit for bit like any other value: the cores emit a fixed
    // qNaN (0xFFC0) and never propagate the input NaN payload.
    if (got == expected) return true;
    g_test_errors++;
    if (g_test_errors <= 30) {
        std::cout << "  [" << test << "] MISMATCH in=0x" << std::hex
                  << std::setw(4) << std::setfill('0') << input
                  << " exp=0x" << std::setw(4) << expected
                  << " got=0x" << std::setw(4) << got
                  << std::dec << std::endl;
    }
    return false;
}

// ---- AXI-Stream Master model ------------------------------------------------
// Follows the protocol: once tvalid+tdata are asserted, they MUST stay stable
// until the handshake completes (tvalid && tready).

struct AxiMaster {
    bool     pending;      // True = we have data asserted on the bus
    uint16_t pend_data;
    uint8_t  pend_user;
    uint16_t pend_expected;
    uint16_t pend_input;   // For debug

    int      next_idx;     // Index of next item to send
    int      count;        // Total items to send

    AxiMaster() : pending(false), next_idx(0), count(0) {}

    // Compute test data for index i
    virtual uint16_t input_data(int i)  const = 0;
    virtual bool     input_base2(int i) const = 0;

    // Try to load next item into pending slot (returns false if nothing to load)
    bool try_load_next() {
        if (pending || next_idx >= count) return false;
        uint16_t raw  = input_data(next_idx);
        bool     b2   = input_base2(next_idx);
        pend_data     = raw;
        pend_user     = b2 ? 1 : 0;
        pend_expected = bf16_exp2_approx(raw, b2);
        pend_input    = raw;
        pending       = true;
        return true;
    }

    // Drive the DUT s_axis signals
    void drive(Vbf16_exp2* dut) const {
        if (pending) {
            dut->s_axis_tvalid = 1;
            dut->s_axis_tdata  = pend_data;
            dut->s_axis_tuser  = pend_user;
        } else {
            dut->s_axis_tvalid = 0;
        }
    }

    // Call after axi_tick(): if input was accepted, push to expected queue
    void on_accepted(TickResult& t, std::deque<Expected>& q) {
        if (t.input_accepted && pending) {
            q.push_back({pend_input, pend_expected});
            pending = false;
            next_idx++;
        }
    }

    bool done() const { return next_idx >= count && !pending; }

    virtual ~AxiMaster() {}
};

// ---- Test 1: Basic continuous flow ------------------------------------------

struct BasicMaster : AxiMaster {
    uint16_t input_data(int i)  const override { return (uint16_t)(i * 257); }
    bool     input_base2(int i) const override { return true; }
};

static bool test_basic_flow(Vbf16_exp2* dut) {
    std::cout << "\n[TEST 1] Basic continuous flow ..." << std::endl;
    reset_dut(dut);
    g_test_errors = 0;

    BasicMaster master;
    master.count = 256;
    std::deque<Expected> q;
    int received = 0;

    for (int cycle = 0; cycle < 300; cycle++) {
        master.try_load_next();
        master.drive(dut);
        dut->m_axis_tready = 1;

        TickResult t = axi_tick(dut);

        if (t.output_consumed && !q.empty()) {
            compare(q.front().expected, t.output_data, q.front().input, "basic");
            q.pop_front();
            received++;
        }
        master.on_accepted(t, q);

        if (master.done() && q.empty()) break;
    }

    bool pass = (received == 256) && q.empty() && (g_test_errors == 0);
    std::cout << "  sent=" << master.next_idx << " received=" << received
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---- Test 2: Backpressure (random m_axis_tready) ----------------------------

struct Bp2Master : AxiMaster {
    uint16_t input_data(int i)  const override { return (uint16_t)(i * 131 + 7); }
    bool     input_base2(int i) const override { return false; }
};

static bool test_backpressure(Vbf16_exp2* dut) {
    std::cout << "\n[TEST 2] Backpressure (random m_axis_tready) ..." << std::endl;
    reset_dut(dut);
    g_test_errors = 0;

    Bp2Master master;
    master.count = 512;
    std::deque<Expected> q;
    int received = 0;

    for (int cycle = 0; cycle < 512 * 8; cycle++) {
        master.try_load_next();
        master.drive(dut);
        dut->m_axis_tready = (prng() % 5 < 2) ? 1 : 0;  // ~40% ready

        TickResult t = axi_tick(dut);

        if (t.output_consumed && !q.empty()) {
            compare(q.front().expected, t.output_data, q.front().input, "backpressure");
            q.pop_front();
            received++;
        }
        master.on_accepted(t, q);

        if (master.done() && q.empty()) break;
    }

    bool pass = (received == 512) && q.empty() && (g_test_errors == 0);
    std::cout << "  sent=" << master.next_idx << " received=" << received
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---- Test 3: Long stall (output must hold) ----------------------------------

static bool test_stall_hold(Vbf16_exp2* dut) {
    std::cout << "\n[TEST 3] Long stall - output value must hold ..." << std::endl;
    reset_dut(dut);

    uint16_t raw = 0x3F00;
    uint16_t expected = bf16_exp2_approx(raw, true);

    // Push one value through (tready=1 so it gets accepted)
    dut->s_axis_tdata  = raw;
    dut->s_axis_tuser  = 1;
    dut->s_axis_tvalid = 1;
    dut->m_axis_tready = 1;
    axi_tick(dut);

    // Stop driving
    dut->s_axis_tvalid = 0;

    // Wait for first valid output (pipeline latency ~7 cycles)
    int warmup = 0;
    for (; warmup < 20; warmup++) {
        dut->m_axis_tready = 1;
        TickResult t = axi_tick(dut);
        if (t.output_consumed) {
            // First valid output - now check it
            bool first_ok = (t.output_data == expected);
            if (!first_ok) {
                std::cout << "  Initial output mismatch: exp=0x" << std::hex
                          << expected << " got=0x" << t.output_data << std::dec << std::endl;
                return false;
            }
            break;
        }
    }

    // Now push another value, let it become valid, then stall
    dut->s_axis_tdata  = 0x4000;
    dut->s_axis_tuser  = 1;
    dut->s_axis_tvalid = 1;
    dut->m_axis_tready = 1;
    axi_tick(dut);
    dut->s_axis_tvalid = 0;

    uint16_t expected2 = bf16_exp2_approx(0x4000, true);
    uint16_t held_data = 0;
    bool got_valid = false;

    // Wait for second output to appear
    for (int i = 0; i < 20; i++) {
        dut->m_axis_tready = 1;
        TickResult t = axi_tick(dut);
        if (t.output_consumed) {
            held_data = t.output_data;
            got_valid = true;
            break;
        }
    }
    if (!got_valid) { std::cout << "  no second output  FAIL" << std::endl; return false; }

    // Now STALL: deassert m_axis_tready for 50 cycles
    // Push another value so m_axis_tvalid stays high
    dut->s_axis_tdata  = 0x4080;
    dut->s_axis_tuser  = 1;
    dut->s_axis_tvalid = 1;
    dut->m_axis_tready = 1;
    axi_tick(dut);
    dut->s_axis_tvalid = 0;

    // Wait for it to appear
    uint16_t stall_data = 0;
    for (int i = 0; i < 20; i++) {
        dut->m_axis_tready = 1;
        dut->eval();
        if (dut->m_axis_tvalid) {
            stall_data = dut->m_axis_tdata;
            break;
        }
        axi_tick(dut);
    }

    // NOW stall for 50 cycles and verify output holds
    dut->m_axis_tready = 0;
    bool held = true;
    bool ready_deasserted = true;
    for (int i = 0; i < 50; i++) {
        dut->eval();
        if (!dut->m_axis_tvalid) { held = false; break; }
        if (dut->m_axis_tdata != stall_data) { held = false; break; }
        if (dut->s_axis_tready != 0) { ready_deasserted = false; }
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }

    bool pass = held && ready_deasserted;
    std::cout << "  output_held=" << held << " ready_deasserted=" << ready_deasserted
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---- Test 4: Input gaps (random s_axis_tvalid) ------------------------------

struct GapMaster : AxiMaster {
    uint16_t input_data(int i)  const override { return (uint16_t)(i * 193 + 11); }
    bool     input_base2(int i) const override { return true; }
};

static bool test_input_gaps(Vbf16_exp2* dut) {
    std::cout << "\n[TEST 4] Input gaps (random s_axis_tvalid) ..." << std::endl;
    reset_dut(dut);
    g_test_errors = 0;

    GapMaster master;
    master.count = 256;
    std::deque<Expected> q;
    int received = 0;
    int spurious = 0;

    dut->m_axis_tready = 1;

    for (int cycle = 0; cycle < 256 * 10; cycle++) {
        // Only try to load new data ~33% of cycles (creates gaps)
        if ((prng() % 3) == 0) master.try_load_next();
        master.drive(dut);
        dut->m_axis_tready = 1;

        TickResult t = axi_tick(dut);

        if (t.output_consumed) {
            if (!q.empty()) {
                compare(q.front().expected, t.output_data, q.front().input, "gaps");
                q.pop_front();
                received++;
            } else {
                spurious++;
            }
        }
        master.on_accepted(t, q);

        if (master.done() && q.empty()) break;
    }

    bool pass = (received == 256) && q.empty() && (spurious == 0) && (g_test_errors == 0);
    std::cout << "  sent=" << master.next_idx << " received=" << received
              << " spurious=" << spurious
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---- Test 5: Protocol - s_axis_tready deasserts under backpressure ----------

static bool test_protocol_ready(Vbf16_exp2* dut) {
    std::cout << "\n[TEST 5] Protocol: s_axis_tready deasserts on stall ..." << std::endl;
    reset_dut(dut);

    // Fill pipeline
    dut->m_axis_tready = 1;
    dut->s_axis_tvalid = 1;
    dut->s_axis_tdata  = 0x3C00;
    dut->s_axis_tuser  = 1;
    for (int i = 0; i < 12; i++) axi_tick(dut);

    // Stall output
    dut->m_axis_tready = 0;

    bool saw_ready_low = false;
    for (int i = 0; i < 20; i++) {
        dut->eval();
        if (dut->m_axis_tvalid && !dut->m_axis_tready) {
            if (dut->s_axis_tready == 0) { saw_ready_low = true; break; }
        }
        dut->clk = 1; dut->eval();
        dut->clk = 0; dut->eval();
    }

    // When stalled: m_axis_tvalid=1,tready=0 => s_axis_tready must be 0
    dut->eval();
    bool ready_correct = true;
    if (dut->m_axis_tvalid && !dut->m_axis_tready) {
        if (dut->s_axis_tready != 0) ready_correct = false;
    }

    bool pass = saw_ready_low && ready_correct;
    std::cout << "  saw_ready_low=" << saw_ready_low
              << " ready_correct=" << ready_correct
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---- Test 6: Combined stress (random valid + random ready) ------------------

struct StressMaster : AxiMaster {
    uint16_t input_data(int i)  const override { return (uint16_t)(i * 67 + 3); }
    bool     input_base2(int i) const override { return (i & 1) != 0; }
};

static bool test_combined_stress(Vbf16_exp2* dut) {
    std::cout << "\n[TEST 6] Combined stress (random valid + random ready) ..." << std::endl;
    reset_dut(dut);
    g_test_errors = 0;

    StressMaster master;
    master.count = 1024;
    std::deque<Expected> q;
    int received = 0;

    for (int cycle = 0; cycle < 1024 * 12; cycle++) {
        // Random decision to load next data (~66% of cycles)
        if ((prng() % 3) != 0) master.try_load_next();
        master.drive(dut);
        dut->m_axis_tready = (prng() % 4 != 0) ? 1 : 0;  // ~75% ready

        TickResult t = axi_tick(dut);

        if (t.output_consumed && !q.empty()) {
            compare(q.front().expected, t.output_data, q.front().input, "stress");
            q.pop_front();
            received++;
        }
        master.on_accepted(t, q);

        if (master.done() && q.empty()) break;
    }

    bool pass = (received == 1024) && q.empty() && (g_test_errors == 0);
    std::cout << "  sent=" << master.next_idx << " received=" << received
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---- Test 7: Full 65536 sweep under random backpressure ---------------------

struct SweepMaster : AxiMaster {
    uint16_t input_data(int i)  const override { return (uint16_t)i; }
    bool     input_base2(int i) const override { return true; }
};

static bool test_full_sweep_backpressure(Vbf16_exp2* dut) {
    std::cout << "\n[TEST 7] Full 65536 sweep with random backpressure ..." << std::endl;
    reset_dut(dut);
    g_test_errors = 0;

    SweepMaster master;
    master.count = 65536;
    std::deque<Expected> q;
    uint64_t received = 0, nan_checked = 0;

    for (int cycle = 0; cycle < 65536 * 6; cycle++) {
        master.try_load_next();
        master.drive(dut);
        dut->m_axis_tready = (prng() % 3 != 0) ? 1 : 0;  // ~66% ready

        TickResult t = axi_tick(dut);

        if (t.output_consumed && !q.empty()) {
            Expected& e = q.front();
            if (is_nan_bf16(e.expected)) {
                nan_checked++;
            }
            if (t.output_data != e.expected) {
                g_test_errors++;
                if (g_test_errors <= 10) {
                    std::cout << "  MISMATCH in=0x" << std::hex
                              << std::setw(4) << std::setfill('0') << e.input
                              << " exp=0x" << std::setw(4) << e.expected
                              << " got=0x" << std::setw(4) << t.output_data
                              << std::dec << std::endl;
                }
            }
            q.pop_front();
            received++;
        }
        master.on_accepted(t, q);

        if (master.done() && q.empty()) break;
    }

    bool pass = (received == 65536) && (g_test_errors == 0);
    std::cout << "  sent=" << master.next_idx << " received=" << received
              << " nan_checked=" << nan_checked << " mismatch=" << g_test_errors
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ---- Main -------------------------------------------------------------------

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vbf16_exp2* dut = new Vbf16_exp2;

    std::cout << "=== AXI-Stream Protocol & Backpressure Test ===" << std::endl;

    int pass_count = 0, fail_count = 0;
    auto run = [&](bool (*fn)(Vbf16_exp2*)) {
        if (fn(dut)) pass_count++; else fail_count++;
    };

    run(test_basic_flow);
    run(test_backpressure);
    run(test_stall_hold);
    run(test_input_gaps);
    run(test_protocol_ready);
    run(test_combined_stress);
    run(test_full_sweep_backpressure);

    delete dut;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << pass_count << " / " << (pass_count + fail_count) << std::endl;
    if (fail_count == 0)
        std::cout << "OVERALL: SUCCESS" << std::endl;
    else
        std::cout << "OVERALL: FAILED (" << fail_count << " test(s) failed)" << std::endl;

    return (fail_count == 0) ? 0 : 1;
}

// Exhaustive Verilator equivalence test for bf16_expe_poly4.
//
// Drives every one of the 65536 BF16 payloads through the AXI-Stream interface
// and compares the RTL output against the C++ model bf16_expe_poly4_approx(),
// which is itself bit-exact with the correctly-rounded full LUT.

#include "bf16_expe_poly4.hpp"
#include "Vbf16_expe_poly4.h"
#include "verilated.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>

static int rtl_latency = 0;

struct Expected {
    uint16_t input;
    uint16_t output;
};

static void tick(Vbf16_expe_poly4* dut) {
    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->eval();
}

int main(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--latency") == 0) rtl_latency = std::atoi(argv[++i]);
    }
    Verilated::commandArgs(argc, argv);
    auto* dut = new Vbf16_expe_poly4;

    dut->clk = 0;
    dut->rst_n = 0;
    dut->s_axis_tvalid = 0;
    dut->m_axis_tready = 1;
    for (int i = 0; i < 4; ++i) tick(dut);
    dut->rst_n = 1;

    std::deque<Expected> queue;
    uint64_t checked = 0;
    uint64_t mismatches = 0;

    auto check = [&]() {
        if (dut->m_axis_tvalid && !queue.empty()) {
            const Expected expected = queue.front();
            queue.pop_front();
            ++checked;
            const uint16_t actual = dut->m_axis_tdata;
            if (actual != expected.output) {
                ++mismatches;
                if (mismatches <= 20) {
                    std::cout << "Mismatch input=0x" << std::hex << std::uppercase
                              << std::setw(4) << std::setfill('0') << expected.input
                              << " expected=0x" << std::setw(4) << expected.output
                              << " actual=0x" << std::setw(4) << actual
                              << std::dec << std::setfill(' ') << '\n';
                }
            }
        }
    };

    for (uint32_t raw = 0; raw <= 0xFFFF; ++raw) {
        const uint16_t input = static_cast<uint16_t>(raw);
        queue.push_back({input, bf16_expe_poly4_approx(input)});
        dut->s_axis_tdata = input;
        dut->s_axis_tvalid = 1;
        tick(dut);
        check();
    }

    dut->s_axis_tvalid = 0;
    for (int i = 0; i < rtl_latency + 4; ++i) {
        tick(dut);
        check();
    }

    std::cout << "Checked: " << checked << "\nMismatches: " << mismatches << '\n'
              << "Table payload: " << bf16_expe_poly4::PAYLOAD_BITS << " bits\n"
              << "DSP slices   : " << bf16_expe_poly4::DSP_SLICES
              << " Horner steps\n"
              << (mismatches == 0 && checked == 65536 ? "PASS\n" : "FAIL\n");
    delete dut;
    return mismatches == 0 && checked == 65536 ? 0 : 1;
}

// =============================================================================
// gen_bf16_expe_cut_tables.cpp
//
// Generates the tables for the cut-point ladder implementation of BF16 exp(x),
// x <= 0.  See src/approximations/bf16_expe_cut.hpp for the architecture.
//
//   t     = (128 + mant) * LOG2E_Q            constant multiplier, P frac bits
//   V     = t >> (P + 7 - e - B)              single variable shift
//   I     = V >> B                            output exponent = -I-1
//   f     = V & (2^B - 1)                     B-bit fraction of -x*log2(e)
//   base  = LIN_BASE(f >> (B-L)) + cand_dev[] candidate mantissa from a 2^L ROM
//   mant  = base + (f <= cut[base+1])         ONE comparison finishes the job
//
// The 128 stored cut points are NOT the exact mantissa breakpoints
//     thr[k] = 1 - log2((127.5 + k)/128)
// They only have to separate the f values that can actually occur, which
// leaves an open interval per cut.  Each cut is placed inside its interval so
// that the deviation from a linear ladder has the smallest possible spread,
// which is what makes CUT_DEV_BITS come out well below B.
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "bf16_expe_lut_table.hpp"

namespace {

constexpr int P = 23;             // fractional bits of the log2(e) constant
constexpr int B = 16;             // fractional bits of f kept in the datapath
constexpr int L = 8;              // index bits of the candidate ROM
constexpr int MANT_COUNT = 128;
constexpr int BIAS = 127;

// round(log2(e) * 2^P) == 0xB8AA3B for P = 23
constexpr uint32_t LOG2E_Q = 12102203u;

constexpr int64_t NEG_INF = std::numeric_limits<int64_t>::min();
constexpr int64_t POS_INF = std::numeric_limits<int64_t>::max();

struct Sample {
    int exp;
    int mant;
    uint32_t f;       // B-bit fraction
    int cnt;          // mantissa count the ladder must produce (0..128)
};

// Linear cut ladder: cut[k] ~ (128-k)/128 in B fractional bits.
inline int64_t lin_cut(int k) { return static_cast<int64_t>(128 - k) << (B - 7); }

// Linear candidate ladder: base[idx] ~ 128 - idx/2^(L-7).
inline int lin_base(int idx) { return 128 - (idx >> (L - 7)); }

inline uint16_t ref_code(int exp, int mant) {
    const int index = (exp - bf16_expe_lut::LUT_MIN_EXP) * MANT_COUNT + mant;
    return bf16_expe_lut::table[index];
}

void emit_signed_array(std::ofstream& out, const char* type, const char* name,
                       const char* count, const std::vector<int>& data,
                       int per_line) {
    out << "static const " << type << ' ' << name << '[' << count << "] = {\n    ";
    for (std::size_t i = 0; i < data.size(); ++i) {
        out << data[i];
        if (i + 1 != data.size()) out << ", ";
        if ((i + 1) % per_line == 0 && i + 1 != data.size()) out << "\n    ";
    }
    out << "\n};\n\n";
}

// Emits one SystemVerilog ROM module with a synchronous/asynchronous read port.
void emit_sv_rom(std::ofstream& out, const std::string& name, int addr_bits,
                 int data_bits, const std::vector<int>& data,
                 const std::string& style, const std::string& comment) {
    out << "// " << comment << '\n'
        << "module " << name << " #(parameter bit REGISTERED = 1'b1) (\n"
        << "    input  logic clk,\n"
        << "    input  logic pipe_en,\n"
        << "    input  logic [" << (addr_bits - 1) << ":0] addr,\n"
        << "    output logic [" << (data_bits - 1) << ":0] data\n"
        << ");\n"
        << "    (* rom_style = \"" << style << "\" *) logic ["
        << (data_bits - 1) << ":0] rom [0:" << (data.size() - 1) << "];\n"
        << "    initial begin\n";

    for (std::size_t i = 0; i < data.size(); ++i) {
        out << "        rom[" << std::setw(3) << std::setfill(' ') << i << "] = "
            << data_bits << "'d" << data[i] << ";\n";
    }

    out << "    end\n"
        << "    generate\n"
        << "        if (REGISTERED) begin : gen_sync\n"
        << "            always_ff @(posedge clk) if (pipe_en) data <= rom[addr];\n"
        << "        end else begin : gen_async\n"
        << "            assign data = rom[addr];\n"
        << "        end\n"
        << "    endgenerate\n"
        << "endmodule : " << name << "\n\n";
}

void write_rtl_rom(int cut_dev_bits, int cand_dev_bits,
                   int64_t cut_dev_bias, int64_t cand_dev_bias,
                   const std::vector<int>& cut_store,
                   const std::vector<int>& cand_store,
                   const std::vector<int>& tail_payload,
                   int tail_exp, int tail_mant_lo, int tail_mant_hi,
                   int zero_exp, int zero_mant_lo,
                   int total_bits) {
    std::ofstream out("src/rtl/bf16_expe_cut_rom.sv");
    if (!out) {
        std::cerr << "Cannot create src/rtl/bf16_expe_cut_rom.sv\n";
        std::exit(1);
    }

    const int tail_addr_bits = 4;   // 11 entries

    out << "// AUTO-GENERATED by gen_bf16_expe_cut_tables.cpp -- do not edit.\n"
        << "// Tables for the cut-point ladder BF16 exp(x) implementation.\n"
        << "//   cut ROM  : " << cut_store.size() << " x " << cut_dev_bits
        << " = " << cut_store.size() * cut_dev_bits << " bits\n"
        << "//   cand ROM : " << cand_store.size() << " x " << cand_dev_bits
        << " = " << cand_store.size() * cand_dev_bits << " bits\n"
        << "//   tail ROM : " << tail_payload.size() << " x 7 = "
        << tail_payload.size() * 7 << " bits\n"
        << "//   total    : " << total_bits << " bits\n\n";

    // ---- geometry package -------------------------------------------------
    out << "package bf16_expe_cut_pkg;\n"
        << "    // Datapath geometry (mirrors bf16_expe_cut_table.hpp).\n"
        << "    localparam int LOG2E_FRAC_BITS = " << P << ";\n"
        << "    localparam int LOG2E_W         = 24;\n"
        << "    localparam logic [23:0] LOG2E_Q = 24'h"
        << std::hex << std::uppercase << LOG2E_Q << std::dec
        << ";  // round(log2(e) * 2**" << P << ")\n"
        << "    localparam int FRAC_BITS       = " << B << ";\n"
        << "    localparam int CAND_INDEX_BITS = " << L << ";\n"
        << "    localparam int PROD_W          = 32;  // 8 x 24\n"
        << "    // Minimum shift is LOG2E_FRAC_BITS + 7 - MAX_EXP - FRAC_BITS.\n"
        << "    localparam int ALIGNED_W       = " << (32 - (P + 7 - bf16_expe_lut::LUT_MAX_EXP - B))
        << ";\n"
        << "    localparam int INT_W           = ALIGNED_W - FRAC_BITS;\n"
        << "    localparam int CUT_DEV_BITS    = " << cut_dev_bits << ";\n"
        << "    localparam int CAND_DEV_BITS   = " << cand_dev_bits << ";\n"
        << "    localparam int CUT_DEV_BIAS    = " << cut_dev_bias << ";\n"
        << "    localparam int CAND_DEV_BIAS   = " << cand_dev_bias << ";\n"
        << "    localparam int CUT_W           = 18;  // signed, holds [-6k, 66k]\n\n"
        << "    // Subnormal tail and flush-to-zero boundary.\n"
        << "    localparam int TAIL_EXP        = " << tail_exp << ";\n"
        << "    localparam int TAIL_MANT_LO    = " << tail_mant_lo << ";\n"
        << "    localparam int TAIL_MANT_HI    = " << tail_mant_hi << ";\n"
        << "    localparam int ZERO_EXP        = " << zero_exp << ";\n"
        << "    localparam int ZERO_MANT_LO    = " << zero_mant_lo << ";\n\n"
        << "    localparam int PAYLOAD_BITS    = " << total_bits << ";\n"
        << "endpackage : bf16_expe_cut_pkg\n\n";

    emit_sv_rom(out, "bf16_expe_cut_cand_rom", L, cand_dev_bits, cand_store,
                "distributed",
                "Candidate mantissa deviation, addressed by the top "
                    + std::to_string(L) + " bits of f.");

    emit_sv_rom(out, "bf16_expe_cut_cut_rom", 7, cut_dev_bits, cut_store,
                "distributed",
                "Cut-point deviation from the linear ladder, addressed by (k-1).");

    emit_sv_rom(out, "bf16_expe_cut_tail_rom", tail_addr_bits, 7, tail_payload,
                "distributed",
                "Subnormal results for exponent " + std::to_string(tail_exp)
                    + ", mantissa " + std::to_string(tail_mant_lo) + ".."
                    + std::to_string(tail_mant_hi) + ".");

    out.close();
    std::cout << "wrote src/rtl/bf16_expe_cut_rom.sv\n";
}

} // namespace

int main() {
    // -------------------------------------------------------------------
    // 1. Split the input space into the ladder path and the subnormal tail.
    // -------------------------------------------------------------------
    std::vector<Sample> samples;
    std::vector<int> tail_payload;      // non-zero subnormal outputs
    int tail_exp = 0, tail_mant_lo = -1, tail_mant_hi = -1;
    int zero_exp = 0, zero_mant_lo = -1;

    for (int exp = bf16_expe_lut::LUT_MIN_EXP; exp <= bf16_expe_lut::LUT_MAX_EXP; ++exp) {
        for (int mant = 0; mant < MANT_COUNT; ++mant) {
            const uint16_t code = ref_code(exp, mant);

            if (code == 0) {                        // flush to zero
                if (zero_mant_lo < 0) { zero_exp = exp; zero_mant_lo = mant; }
                continue;
            }
            if (code < 128) {                       // subnormal output
                if (tail_mant_lo < 0) { tail_exp = exp; tail_mant_lo = mant; }
                tail_mant_hi = mant;
                tail_payload.push_back(code);
                continue;
            }

            // ---- normal output: replay the integer datapath ----
            const uint64_t prod = static_cast<uint64_t>(128 + mant) * LOG2E_Q;
            const int shift = P + 7 - exp - B;
            const uint64_t V = prod >> shift;
            const int I = static_cast<int>(V >> B);
            const uint32_t f = static_cast<uint32_t>(V & ((1u << B) - 1));

            const int e_out_ref = (code >> 7) - BIAS;
            const int m_out_ref = code & 0x7F;
            const int e_out = -I - 1;

            int cnt;
            if (e_out == e_out_ref) {
                cnt = m_out_ref;
            } else if (e_out == e_out_ref - 1 && m_out_ref == 0) {
                cnt = 128;                          // mantissa overflows to 2.0
            } else {
                std::cerr << "Integer exponent unrecoverable at exp=" << exp
                          << " mant=" << mant << '\n';
                return 1;
            }
            samples.push_back({exp, mant, f, cnt});
        }
    }

    std::cout << "ladder inputs      : " << samples.size() << '\n'
              << "subnormal tail     : " << tail_payload.size()
              << " (exp " << tail_exp << ", mant " << tail_mant_lo
              << ".." << tail_mant_hi << ")\n"
              << "flush-to-zero from : exp " << zero_exp
              << ", mant " << zero_mant_lo << '\n';

    // -------------------------------------------------------------------
    // 2. Feasible interval for every cut point.
    //    cut[k] must satisfy:  f <= cut[k]  <=>  cnt >= k
    // -------------------------------------------------------------------
    std::vector<int64_t> lo(129, NEG_INF);   // max f that needs cnt >= k
    std::vector<int64_t> hi(129, POS_INF);   // min f that needs cnt <  k

    for (const Sample& s : samples) {
        const int64_t f = s.f;
        for (int k = 1; k <= 128; ++k) {
            if (s.cnt >= k) { if (f > lo[k]) lo[k] = f; }
            else            { if (f < hi[k]) hi[k] = f; }
        }
    }
    for (int k = 1; k <= 128; ++k) {
        if (lo[k] != NEG_INF && hi[k] != POS_INF && lo[k] >= hi[k]) {
            std::cerr << "Cut " << k << " is not separable\n";
            return 1;
        }
    }

    // Admissible range of cut[k] expressed as deviation from the linear ladder.
    std::vector<int64_t> dlo(129), dhi(129);
    for (int k = 1; k <= 128; ++k) {
        const int64_t l = (lo[k] == NEG_INF) ? 0 : lo[k];
        const int64_t h = (hi[k] == POS_INF) ? ((1 << B) - 1) : hi[k] - 1;
        dlo[k] = l - lin_cut(k);
        dhi[k] = h - lin_cut(k);
    }

    // Pick every cut so that the spread of the deviations is minimal:
    // the optimum is max_k dlo[k] - min_k dhi[k], reached by clamping the
    // common target max_k dlo[k] into each admissible interval.
    int64_t target = NEG_INF;
    for (int k = 1; k <= 128; ++k) target = std::max(target, dlo[k]);

    std::vector<int> cut_dev(128);
    int64_t dev_min = POS_INF, dev_max = NEG_INF;
    for (int k = 1; k <= 128; ++k) {
        const int64_t d = std::min(std::max(target, dlo[k]), dhi[k]);
        cut_dev[k - 1] = static_cast<int>(d);
        dev_min = std::min(dev_min, d);
        dev_max = std::max(dev_max, d);
    }

    const int64_t cut_span = dev_max - dev_min;
    int cut_dev_bits = 1;
    while ((int64_t(1) << cut_dev_bits) <= cut_span) ++cut_dev_bits;

    std::vector<int> cut_store(128);
    for (int k = 0; k < 128; ++k) cut_store[k] = static_cast<int>(cut_dev[k] - dev_min);

    std::cout << "cut deviation      : " << dev_min << " .. " << dev_max
              << "  -> " << cut_dev_bits << " bits/entry\n";

    // -------------------------------------------------------------------
    // 3. Candidate ROM: top L bits of f give the mantissa up to +1.
    // -------------------------------------------------------------------
    const int buckets = 1 << L;
    std::vector<int> bucket_lo(buckets, 1000), bucket_hi(buckets, -1);
    for (const Sample& s : samples) {
        const int idx = static_cast<int>(s.f >> (B - L));
        bucket_lo[idx] = std::min(bucket_lo[idx], s.cnt);
        bucket_hi[idx] = std::max(bucket_hi[idx], s.cnt);
    }

    int worst_spread = 0;
    for (int i = 0; i < buckets; ++i) {
        if (bucket_hi[i] < 0) continue;             // bucket never occurs
        worst_spread = std::max(worst_spread, bucket_hi[i] - bucket_lo[i]);
    }
    if (worst_spread > 1) {
        std::cerr << "Candidate ROM needs " << worst_spread
                  << " corrections, expected at most 1\n";
        return 1;
    }

    std::vector<int> cand_dev(buckets);
    int64_t cand_min = POS_INF, cand_max = NEG_INF;
    for (int i = 0; i < buckets; ++i) {
        const int base = (bucket_hi[i] < 0) ? lin_base(i) : bucket_lo[i];
        cand_dev[i] = base - lin_base(i);
        cand_min = std::min<int64_t>(cand_min, cand_dev[i]);
        cand_max = std::max<int64_t>(cand_max, cand_dev[i]);
    }

    const int64_t cand_span = cand_max - cand_min;
    int cand_dev_bits = 1;
    while ((int64_t(1) << cand_dev_bits) <= cand_span) ++cand_dev_bits;

    std::vector<int> cand_store(buckets);
    for (int i = 0; i < buckets; ++i)
        cand_store[i] = static_cast<int>(cand_dev[i] - cand_min);

    std::cout << "candidate deviation: " << cand_min << " .. " << cand_max
              << "  -> " << cand_dev_bits << " bits/entry\n";

    // -------------------------------------------------------------------
    // 4. Emit the header.
    // -------------------------------------------------------------------
    std::ofstream out("src/approximations/bf16_expe_cut_table.hpp");
    if (!out) {
        std::cerr << "Cannot create src/approximations/bf16_expe_cut_table.hpp\n";
        return 1;
    }

    const int cut_bits = 128 * cut_dev_bits;
    const int cand_bits = buckets * cand_dev_bits;
    const int tail_bits = static_cast<int>(tail_payload.size()) * 7;
    const int total_bits = cut_bits + cand_bits + tail_bits;

    out << "// AUTO-GENERATED by gen_bf16_expe_cut_tables.cpp -- do not edit.\n"
        << "#ifndef BF16_EXPE_CUT_TABLE_HPP\n"
        << "#define BF16_EXPE_CUT_TABLE_HPP\n\n"
        << "#include <cstdint>\n\n"
        << "namespace bf16_expe_cut {\n\n"
        << "// ---- datapath geometry ----\n"
        << "constexpr int LOG2E_FRAC_BITS = " << P << ";\n"
        << "constexpr uint32_t LOG2E_Q    = 0x"
        << std::hex << std::uppercase << LOG2E_Q << std::dec << "u;  // round(log2(e) * 2^"
        << P << ")\n"
        << "constexpr int FRAC_BITS       = " << B << ";\n"
        << "constexpr int CAND_INDEX_BITS = " << L << ";\n"
        << "constexpr int CAND_COUNT      = " << buckets << ";\n"
        << "constexpr int CUT_COUNT       = 128;\n"
        << "constexpr int MIN_EXP         = " << bf16_expe_lut::LUT_MIN_EXP << ";\n"
        << "constexpr int MAX_EXP         = " << bf16_expe_lut::LUT_MAX_EXP << ";\n\n"
        << "// ---- storage widths (what actually costs FPGA/ASIC area) ----\n"
        << "constexpr int CUT_DEV_BITS    = " << cut_dev_bits << ";\n"
        << "constexpr int CAND_DEV_BITS   = " << cand_dev_bits << ";\n"
        << "constexpr int CUT_DEV_BIAS    = " << dev_min << ";\n"
        << "constexpr int CAND_DEV_BIAS   = " << cand_min << ";\n"
        << "constexpr int PAYLOAD_BITS    = " << total_bits
        << ";  // " << cut_bits << " + " << cand_bits << " + " << tail_bits << "\n\n"
        << "// ---- subnormal tail ----\n"
        << "constexpr int TAIL_EXP        = " << tail_exp << ";\n"
        << "constexpr int TAIL_MANT_LO    = " << tail_mant_lo << ";\n"
        << "constexpr int TAIL_MANT_HI    = " << tail_mant_hi << ";\n"
        << "constexpr int TAIL_COUNT      = " << tail_payload.size() << ";\n"
        << "constexpr int ZERO_EXP        = " << zero_exp << ";\n"
        << "constexpr int ZERO_MANT_LO    = " << zero_mant_lo << ";\n\n";

    out << "// cut[k] = ((128-k) << (FRAC_BITS-7)) + CUT_DEV_BIAS + cut_dev[k-1]\n";
    emit_signed_array(out, "uint16_t", "cut_dev", "CUT_COUNT", cut_store, 12);

    out << "// base[idx] = (128 - (idx >> (CAND_INDEX_BITS-7)))\n"
        << "//             + CAND_DEV_BIAS + cand_dev[idx]\n";
    emit_signed_array(out, "uint8_t", "cand_dev", "CAND_COUNT", cand_store, 24);

    out << "// BF16 payloads of the subnormal results, mantissa TAIL_MANT_LO..HI\n";
    emit_signed_array(out, "uint8_t", "tail_payload", "TAIL_COUNT", tail_payload, 12);

    out << "} // namespace bf16_expe_cut\n\n#endif\n";
    out.close();

    write_rtl_rom(cut_dev_bits, cand_dev_bits, dev_min, cand_min,
                  cut_store, cand_store, tail_payload,
                  tail_exp, tail_mant_lo, tail_mant_hi,
                  zero_exp, zero_mant_lo, total_bits);

    // -------------------------------------------------------------------
    // 5. Self-check: replay the exact datapath over the whole ladder set.
    // -------------------------------------------------------------------
    std::vector<int64_t> cut(130);
    for (int k = 1; k <= 128; ++k) cut[k] = lin_cut(k) + cut_dev[k - 1];
    cut[129] = -1;                                  // nothing is <= -1

    int mismatches = 0;
    for (const Sample& s : samples) {
        const int idx = static_cast<int>(s.f >> (B - L));
        const int base = lin_base(idx) + cand_dev[idx];
        const int cnt = base + ((static_cast<int64_t>(s.f) <= cut[base + 1]) ? 1 : 0);
        if (cnt != s.cnt) ++mismatches;
    }

    std::cout << "generator self-check: " << mismatches << " mismatches\n"
              << "payload            : " << total_bits << " bits ("
              << cut_bits << " cuts + " << cand_bits << " candidates + "
              << tail_bits << " tail)\n"
              << "wrote src/approximations/bf16_expe_cut_table.hpp\n";

    return mismatches == 0 ? 0 : 1;
}

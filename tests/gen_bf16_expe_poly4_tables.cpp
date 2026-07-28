// =============================================================================
// gen_bf16_expe_poly4_tables.cpp
//
// Generates the coefficients for the DEGREE-4 MINIMAX/HORNER implementation of
// BF16 exp(x), x <= 0.  See src/approximations/bf16_expe_poly4.hpp.
//
// Front end is identical to the cut-point ladder:
//   t     = (128 + mant) * LOG2E_Q          constant multiplier, P frac bits
//   V     = t >> (P + 7 - e - B)            single variable shift
//   I     = V >> B                          output exponent = -I-1
//   f     = V & (2^B - 1)                   B-bit fraction of -x*log2(e)
//
// Back end evaluates P(f) ~= 2^(1-f) with Horner and rounds to 7 bits:
//   acc   = c4
//   acc   = (acc*f + C_k) >> B              for k = 3, 2, 1, 0
//   mant  = (acc - ONE + HALF) >> (ACC_FRAC - 7)
//
// DSP MAPPING (this is the point of the variant)
// ----------------------------------------------
// A Xilinx DSP48E1 computes P = A*B + C with A 25-bit signed, B 18-bit signed.
// Choosing ACC_W = 25 and B = 17 makes ONE Horner step ONE DSP slice:
//     A <- acc (25 b signed)
//     B <- f   (17 b unsigned, fits an 18-bit signed port)
//     C <- (c_k << (ACC_FRAC - COEF_FRAC + B)) + (1 << (B-1))
// The rounding constant is folded into C, so round-to-nearest at every Horner
// step is FREE.  Degree 4 is therefore a 4-slice DSP cascade (PCIN chained).
//
// COEFFICIENT SEARCH
// ------------------
// Minimax is the wrong objective here.  We do not need |P(f) - 2^(1-f)| to be
// small, we need
//     round(128 * (P(f) - 1)) == correctly rounded mantissa code
// for the ~1959 truncated f values that actually occur.  That is a set of
// interval constraints on the INTEGER accumulator:
//     lo_i <= acc0(f_i) <= hi_i
// so the generator maximizes the worst-case slack directly, on the exact
// integer datapath, by multi-scale coordinate descent seeded from a least
// squares fit.  Truncation and coefficient quantization are therefore part of
// the objective rather than an error added on top of it.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

#include "bf16_expe_lut_table.hpp"

namespace {

constexpr int P_FRAC = 23;        // fractional bits of the log2(e) constant
constexpr int B = 17;             // fractional bits of f (DSP48 B port)
constexpr int DEG = 4;            // polynomial degree -> DSP slice count
constexpr int NCOEF = DEG + 1;
constexpr int ACC_FRAC = 22;      // fractional bits of the Horner accumulator
constexpr int ACC_W = 25;         // DSP48 A port width
constexpr int MANT_COUNT = 128;
constexpr int MIN_EXP = -9;
constexpr int MAX_EXP = 7;

constexpr uint32_t LOG2E_Q = 12102203u;   // round(log2(e) * 2^23) == 0xB8AA3B

constexpr int64_t ONE = int64_t{1} << ACC_FRAC;
constexpr int64_t HALF = int64_t{1} << (ACC_FRAC - 8);
constexpr int ROUND_SHIFT = ACC_FRAC - 7;
constexpr int64_t STEP_ROUND = int64_t{1} << (B - 1);   // folded into C

struct Sample {
    uint32_t f;        // B-bit fraction
    int required;      // mantissa code the back end must produce (0..128)
    int64_t lo, hi;    // admissible accumulator window
};

// ---------------------------------------------------------------------------
// Reference: the full LUT stores correctly rounded exp(x) for every BF16 input.
// ---------------------------------------------------------------------------
inline uint16_t ref_code(int exp, int mant) {
    const int index = (exp - bf16_expe_lut::LUT_MIN_EXP) * MANT_COUNT + mant;
    return bf16_expe_lut::table[index];
}

// ---------------------------------------------------------------------------
// Exact integer Horner, bit-for-bit what the RTL and the C++ model will do.
// ---------------------------------------------------------------------------
inline int64_t horner(const int64_t* cq, int coef_frac, uint32_t f) {
    const int sh = ACC_FRAC - coef_frac;
    int64_t acc = cq[DEG] << sh;
    for (int k = DEG - 1; k >= 0; --k) {
        const int64_t c_term = (cq[k] << sh) << B;
        acc = (acc * static_cast<int64_t>(f) + c_term + STEP_ROUND) >> B;
    }
    return acc;
}

// Worst-case distance to a rounding boundary, in accumulator LSBs.
// Negative means at least one input is decoded to the wrong mantissa code.
int64_t worst_slack(const std::vector<Sample>& s, const int64_t* cq, int coef_frac) {
    int64_t worst = std::numeric_limits<int64_t>::max();
    for (const Sample& sm : s) {
        const int64_t acc = horner(cq, coef_frac, sm.f);
        worst = std::min(worst, std::min(acc - sm.lo, sm.hi - acc));
        if (worst < -(int64_t{1} << 40)) break;   // hopeless, stop early
    }
    return worst;
}

// ---------------------------------------------------------------------------
// Least squares seed: solve the normal equations in long double.
// ---------------------------------------------------------------------------
std::vector<long double> least_squares(const std::vector<Sample>& s) {
    long double a[NCOEF][NCOEF + 1] = {};
    for (const Sample& sm : s) {
        const long double x = static_cast<long double>(sm.f) / (1u << B);
        const long double y = powl(2.0L, 1.0L - x);
        long double pw[NCOEF];
        pw[0] = 1.0L;
        for (int i = 1; i < NCOEF; ++i) pw[i] = pw[i - 1] * x;
        for (int i = 0; i < NCOEF; ++i) {
            for (int j = 0; j < NCOEF; ++j) a[i][j] += pw[i] * pw[j];
            a[i][NCOEF] += pw[i] * y;
        }
    }
    for (int col = 0; col < NCOEF; ++col) {          // Gauss, partial pivoting
        int piv = col;
        for (int r = col + 1; r < NCOEF; ++r)
            if (fabsl(a[r][col]) > fabsl(a[piv][col])) piv = r;
        for (int c = 0; c <= NCOEF; ++c) std::swap(a[col][c], a[piv][c]);
        for (int r = 0; r < NCOEF; ++r) {
            if (r == col) continue;
            const long double factor = a[r][col] / a[col][col];
            for (int c = col; c <= NCOEF; ++c) a[r][c] -= factor * a[col][c];
        }
    }
    std::vector<long double> out(NCOEF);
    for (int i = 0; i < NCOEF; ++i) out[i] = a[i][NCOEF] / a[i][i];
    return out;
}

// ---------------------------------------------------------------------------
// Multi-scale integer coordinate descent maximizing the worst-case slack.
// Runs from several seeds and keeps the best, because the objective is
// piecewise constant and plain descent from a single seed gets stuck.
// ---------------------------------------------------------------------------
int64_t refine(const std::vector<Sample>& s, int coef_frac,
               const int64_t* start, int64_t* best) {
    std::copy(start, start + NCOEF, best);
    int64_t best_slack = worst_slack(s, best, coef_frac);
    int64_t trial[NCOEF];

    for (int step = 1 << 12; step >= 1; step >>= 1) {
        bool improved = true;
        while (improved) {
            improved = false;
            for (int i = 0; i < NCOEF; ++i) {
                for (int dir = -1; dir <= 1; dir += 2) {
                    std::copy(best, best + NCOEF, trial);
                    trial[i] += dir * step;
                    const int64_t slack = worst_slack(s, trial, coef_frac);
                    if (slack > best_slack) {
                        best_slack = slack;
                        std::copy(trial, trial + NCOEF, best);
                        improved = true;
                    }
                }
            }
        }
    }
    return best_slack;
}

int64_t search_coeffs(const std::vector<Sample>& s, int coef_frac,
                      const std::vector<long double>& seed,
                      const int64_t* carried, bool have_carried,
                      int64_t* best) {
    int64_t start[NCOEF];
    for (int i = 0; i < NCOEF; ++i)
        start[i] = llroundl(seed[i] * powl(2.0L, coef_frac));

    int64_t best_slack = refine(s, coef_frac, start, best);

    if (have_carried) {   // previous width's answer, rescaled one bit up
        int64_t alt[NCOEF], out[NCOEF];
        for (int i = 0; i < NCOEF; ++i) alt[i] = carried[i] * 2;
        const int64_t slack = refine(s, coef_frac, alt, out);
        if (slack > best_slack) {
            best_slack = slack;
            std::copy(out, out + NCOEF, best);
        }
    }
    return best_slack;
}

// ---------------------------------------------------------------------------
// Emitters
// ---------------------------------------------------------------------------
void write_cpp_header(int coef_w, int coef_frac, const int64_t* cq,
                      const std::vector<int>& tail, int tail_lo, int tail_hi,
                      int zero_lo, int payload_bits, int64_t slack) {
    std::ofstream out("src/approximations/bf16_expe_poly4_table.hpp");
    if (!out) {
        std::cerr << "Cannot create src/approximations/bf16_expe_poly4_table.hpp\n";
        std::exit(1);
    }
    out << "// AUTO-GENERATED by gen_bf16_expe_poly4_tables.cpp -- do not edit.\n"
        << "#ifndef BF16_EXPE_POLY4_TABLE_HPP\n"
        << "#define BF16_EXPE_POLY4_TABLE_HPP\n\n"
        << "#include <cstdint>\n\n"
        << "namespace bf16_expe_poly4 {\n\n"
        << "// ---- datapath geometry ----\n"
        << "constexpr int LOG2E_FRAC_BITS = " << P_FRAC << ";\n"
        << "constexpr uint32_t LOG2E_Q    = 0x" << std::hex << std::uppercase
        << LOG2E_Q << std::dec << "u;  // round(log2(e) * 2^" << P_FRAC << ")\n"
        << "constexpr int FRAC_BITS       = " << B << ";   // DSP48 B port\n"
        << "constexpr int DEGREE          = " << DEG << ";\n"
        << "constexpr int COEF_COUNT      = " << NCOEF << ";\n"
        << "constexpr int ACC_FRAC        = " << ACC_FRAC << ";\n"
        << "constexpr int ACC_W           = " << ACC_W << ";   // DSP48 A port\n"
        << "constexpr int COEF_W          = " << coef_w << ";\n"
        << "constexpr int COEF_FRAC       = " << coef_frac << ";\n"
        << "constexpr int ROUND_SHIFT     = " << ROUND_SHIFT << ";\n"
        << "constexpr int MIN_EXP         = " << MIN_EXP << ";\n"
        << "constexpr int MAX_EXP         = " << MAX_EXP << ";\n"
        << "constexpr int PAYLOAD_BITS    = " << payload_bits << ";  // "
        << (coef_w * NCOEF) << " + " << (static_cast<int>(tail.size()) * 7) << "\n"
        << "constexpr int DSP_SLICES      = " << DEG << ";   // one per Horner step\n\n"
        << "// Worst-case distance to a rounding boundary over all inputs,\n"
        << "// in accumulator LSBs (must be >= 0 for bit-exactness).\n"
        << "constexpr int64_t WORST_SLACK = " << slack << ";\n\n"
        << "// ---- subnormal tail ----\n"
        << "constexpr int TAIL_EXP        = 6;\n"
        << "constexpr int TAIL_MANT_LO    = " << tail_lo << ";\n"
        << "constexpr int TAIL_MANT_HI    = " << tail_hi << ";\n"
        << "constexpr int TAIL_COUNT      = " << tail.size() << ";\n"
        << "constexpr int ZERO_EXP        = 6;\n"
        << "constexpr int ZERO_MANT_LO    = " << zero_lo << ";\n\n"
        << "// P(f) = ((((c4*f + c3)*f + c2)*f + c1)*f + c0), coefficients scaled\n"
        << "// by 2^COEF_FRAC.  Ascending order.\n"
        << "static const int32_t coef[COEF_COUNT] = {\n";
    for (int i = 0; i < NCOEF; ++i) {
        out << "    " << cq[i] << (i + 1 == NCOEF ? "" : ",")
            << "  // c" << i << " = " << std::fixed << std::setprecision(12)
            << (static_cast<double>(cq[i]) / std::ldexp(1.0, coef_frac)) << '\n';
    }
    out << "};\n\n"
        << "// BF16 payloads of the subnormal results, mantissa TAIL_MANT_LO..HI\n"
        << "static const uint8_t tail_payload[TAIL_COUNT] = {\n    ";
    for (std::size_t i = 0; i < tail.size(); ++i)
        out << tail[i] << (i + 1 == tail.size() ? "" : ", ");
    out << "\n};\n\n"
        << "} // namespace bf16_expe_poly4\n\n"
        << "#endif\n";
}

void write_sv_package(int coef_w, int coef_frac, const int64_t* cq,
                      const std::vector<int>& tail, int tail_lo, int tail_hi,
                      int zero_lo, int payload_bits) {
    std::ofstream out("src/rtl/bf16_expe_poly4_rom.sv");
    if (!out) {
        std::cerr << "Cannot create src/rtl/bf16_expe_poly4_rom.sv\n";
        std::exit(1);
    }
    const int prod_w = ACC_W + B + 1;

    out << "// AUTO-GENERATED by gen_bf16_expe_poly4_tables.cpp -- do not edit.\n"
        << "// Coefficients for the degree-" << DEG
        << " Horner BF16 exp(x) implementation.\n"
        << "//   coefficients : " << NCOEF << " x " << coef_w << " = "
        << (NCOEF * coef_w) << " bits\n"
        << "//   tail ROM     : " << tail.size() << " x 7 = "
        << (tail.size() * 7) << " bits\n"
        << "//   total        : " << payload_bits << " bits\n"
        << "//\n"
        << "// Each Horner step maps to ONE DSP48E1: A = acc (" << ACC_W
        << " b signed),\n"
        << "// B = f (" << B << " b unsigned), C = coefficient pre-shifted into"
        << " the product\n"
        << "// scale with the rounding constant folded in.\n\n"
        << "/* verilator lint_off DECLFILENAME */\n"
        << "package bf16_expe_poly4_pkg;\n"
        << "    localparam int LOG2E_FRAC_BITS = " << P_FRAC << ";\n"
        << "    localparam logic [23:0] LOG2E_Q = 24'h" << std::hex
        << std::uppercase << LOG2E_Q << std::dec << ";\n"
        << "    localparam int FRAC_BITS   = " << B << ";\n"
        << "    localparam int DEGREE      = " << DEG << ";\n"
        << "    localparam int ACC_FRAC    = " << ACC_FRAC << ";\n"
        << "    localparam int ACC_W       = " << ACC_W << ";\n"
        << "    localparam int COEF_W      = " << coef_w << ";\n"
        << "    localparam int COEF_FRAC   = " << coef_frac << ";\n"
        << "    localparam int PROD_W      = " << prod_w << ";\n"
        << "    localparam int ROUND_SHIFT = " << ROUND_SHIFT << ";\n"
        << "    // Named CONST_MULT_W, not MULT_W: bf16_exp2_pkg already uses that name.\n"
        << "    localparam int CONST_MULT_W = 32;   // 8 x 24 constant multiply\n"
        << "    // Minimum alignment shift is LOG2E_FRAC_BITS + 7 - MAX_EXP - FRAC_BITS.\n"
        << "    localparam int ALIGNED_W   = " << (P_FRAC + 7 - MAX_EXP - B + B + 3)
        << ";\n"
        << "    localparam int INT_W       = ALIGNED_W - FRAC_BITS;\n\n"
        << "    // Rounding constant folded into every DSP C port (free).\n"
        << "    localparam logic signed [PROD_W-1:0] STEP_ROUND = "
        << "PROD_W'(" << STEP_ROUND << ");\n\n"
        << "    // Coefficients, ascending order, scaled by 2^COEF_FRAC.\n";
    for (int i = 0; i < NCOEF; ++i) {
        out << "    localparam logic signed [COEF_W-1:0] POLY_C" << i
            << " = COEF_W'(" << cq[i] << ");   // "
            << std::fixed << std::setprecision(12)
            << (static_cast<double>(cq[i]) / std::ldexp(1.0, coef_frac)) << '\n';
    }
    out << "\n    // Subnormal tail and flush-to-zero boundary.\n"
        << "    localparam int TAIL_EXP     = 6;\n"
        << "    localparam int TAIL_MANT_LO = " << tail_lo << ";\n"
        << "    localparam int TAIL_MANT_HI = " << tail_hi << ";\n"
        << "    localparam int ZERO_EXP     = 6;\n"
        << "    localparam int ZERO_MANT_LO = " << zero_lo << ";\n"
        << "    localparam int PAYLOAD_BITS = " << payload_bits << ";\n"
        << "endpackage : bf16_expe_poly4_pkg\n"
        << "/* verilator lint_on DECLFILENAME */\n\n"
        << "// Subnormal results for exponent 6, mantissa " << tail_lo << ".."
        << tail_hi << ".\n"
        << "module bf16_expe_poly4_tail_rom #(parameter bit REGISTERED = 1'b1) (\n"
        << "    input  logic clk,\n"
        << "    input  logic pipe_en,\n"
        << "    input  logic [3:0] addr,\n"
        << "    output logic [6:0] data\n"
        << ");\n"
        << "    (* rom_style = \"distributed\" *) logic [6:0] rom [0:"
        << (tail.size() - 1) << "];\n"
        << "    initial begin\n";
    for (std::size_t i = 0; i < tail.size(); ++i)
        out << "        rom[" << std::setw(3) << i << "] = 7'd" << tail[i] << ";\n";
    out << "    end\n"
        << "    generate\n"
        << "        if (REGISTERED) begin : gen_sync\n"
        << "            always_ff @(posedge clk) if (pipe_en) data <= rom[addr];\n"
        << "        end else begin : gen_async\n"
        << "            assign data = rom[addr];\n"
        << "        end\n"
        << "    endgenerate\n"
        << "endmodule : bf16_expe_poly4_tail_rom\n";
}

} // namespace

int main() {
    // -----------------------------------------------------------------------
    // Classify every input: ladder, subnormal tail, or flush to zero.
    // -----------------------------------------------------------------------
    int tail_lo = -1, tail_hi = -1, zero_lo = -1;
    for (int mant = 0; mant < MANT_COUNT; ++mant) {
        const uint16_t ref = ref_code(6, mant);
        const int biased = (ref >> 7) & 0xFF;
        if (ref == 0) {
            if (zero_lo < 0) zero_lo = mant;
        } else if (biased == 0) {
            if (tail_lo < 0) tail_lo = mant;
            tail_hi = mant;
        }
    }
    std::vector<int> tail;
    for (int m = tail_lo; m <= tail_hi; ++m) tail.push_back(ref_code(6, m) & 0x7F);

    std::cout << "subnormal tail : exp 6, mantissa " << tail_lo << ".." << tail_hi
              << " (" << tail.size() << " entries)\n"
              << "flush to zero  : exp 6, mantissa >= " << zero_lo << "\n\n";

    // -----------------------------------------------------------------------
    // Build the constraint set on the exact integer datapath.
    // -----------------------------------------------------------------------
    std::map<uint32_t, int> by_f;
    int total_ladder = 0;

    for (int exp = MIN_EXP; exp <= MAX_EXP; ++exp) {
        for (int mant = 0; mant < MANT_COUNT; ++mant) {
            if (exp > 6 || (exp == 6 && mant >= zero_lo)) continue;
            if (exp == 6 && mant >= tail_lo) continue;

            const uint64_t product =
                static_cast<uint64_t>(128 + mant) * LOG2E_Q;
            const int shift = P_FRAC + 7 - exp - B;
            const uint64_t aligned = product >> shift;
            const int int_part = static_cast<int>(aligned >> B);
            const uint32_t f =
                static_cast<uint32_t>(aligned & ((1u << B) - 1));

            const uint16_t ref = ref_code(exp, mant);
            const int biased = (ref >> 7) & 0xFF;
            const int ref_mant = ref & 0x7F;

            int required;
            if (biased == 126 - int_part) {
                required = ref_mant;
            } else if (biased == 127 - int_part && ref_mant == 0) {
                required = 128;         // mantissa rounded up to 2.0
            } else {
                std::cerr << "Exponent mismatch at exp=" << exp
                          << " mant=" << mant << " I=" << int_part
                          << " ref=0x" << std::hex << ref << std::dec << '\n';
                return 1;
            }

            ++total_ladder;
            auto it = by_f.find(f);
            if (it == by_f.end()) {
                by_f.emplace(f, required);
            } else if (it->second != required) {
                std::cerr << "Conflicting requirement for f=" << f << '\n';
                return 1;
            }
        }
    }

    std::vector<Sample> samples;
    samples.reserve(by_f.size());
    for (const auto& kv : by_f) {
        Sample s;
        s.f = kv.first;
        s.required = kv.second;
        s.lo = ONE - HALF + (static_cast<int64_t>(s.required) << ROUND_SHIFT);
        s.hi = ONE - HALF +
               (static_cast<int64_t>(s.required + 1) << ROUND_SHIFT) - 1;
        samples.push_back(s);
    }

    std::cout << "ladder inputs  : " << total_ladder << '\n'
              << "distinct f     : " << samples.size() << "  (f width " << B
              << " bits)\n\n";

    // -----------------------------------------------------------------------
    // Search for the narrowest coefficients that decode every input correctly.
    // -----------------------------------------------------------------------
    const std::vector<long double> seed = least_squares(samples);
    std::cout << "least squares seed:\n";
    for (int i = 0; i < NCOEF; ++i)
        std::cout << "    c" << i << " = " << std::fixed << std::setprecision(12)
                  << static_cast<double>(seed[i]) << '\n';
    std::cout << "\ncoefficient width sweep (slack in accumulator LSBs):\n"
              << " width | frac | worst slack | verdict\n"
              << "-------+------+-------------+-----------\n";

    int64_t cq[NCOEF] = {};
    int chosen_w = -1, chosen_frac = -1;
    int64_t chosen_slack = 0;

    int64_t carried[NCOEF] = {};
    bool have_carried = false;

    for (int coef_w = 12; coef_w <= ACC_FRAC + 2; ++coef_w) {
        const int coef_frac = coef_w - 2;      // sign + 1 integer bit
        int64_t trial[NCOEF];
        const int64_t slack =
            search_coeffs(samples, coef_frac, seed, carried, have_carried, trial);
        const bool ok = slack >= 0;
        std::cout << std::setw(6) << coef_w << " | " << std::setw(4) << coef_frac
                  << " | " << std::setw(11) << slack << " | "
                  << (ok ? "bit-exact" : "fails") << '\n';

        std::copy(trial, trial + NCOEF, carried);
        have_carried = true;

        if (ok && chosen_w < 0) {
            std::copy(trial, trial + NCOEF, cq);
            chosen_w = coef_w;
            chosen_frac = coef_frac;
            chosen_slack = slack;
        }
    }

    if (chosen_w < 0) {
        std::cerr << "\nNo bit-exact coefficient set found.\n";
        return 1;
    }

    std::cout << "\nchosen: " << chosen_w << "-bit coefficients ("
              << chosen_frac << " fractional), worst slack " << chosen_slack
              << " of " << (int64_t{1} << (ROUND_SHIFT - 1)) << " LSBs\n";
    for (int i = 0; i < NCOEF; ++i)
        std::cout << "    c" << i << " = " << std::setw(9) << cq[i] << "  = "
                  << std::fixed << std::setprecision(12)
                  << (static_cast<double>(cq[i]) / std::ldexp(1.0, chosen_frac))
                  << '\n';

    // -----------------------------------------------------------------------
    // Replay the complete datapath as a self-check before emitting anything.
    // -----------------------------------------------------------------------
    int mismatches = 0;
    for (int exp = MIN_EXP; exp <= MAX_EXP; ++exp) {
        for (int mant = 0; mant < MANT_COUNT; ++mant) {
            if (exp > 6 || (exp == 6 && mant >= zero_lo)) continue;
            if (exp == 6 && mant >= tail_lo) continue;

            const uint64_t product = static_cast<uint64_t>(128 + mant) * LOG2E_Q;
            const uint64_t aligned = product >> (P_FRAC + 7 - exp - B);
            const int int_part = static_cast<int>(aligned >> B);
            const uint32_t f = static_cast<uint32_t>(aligned & ((1u << B) - 1));

            const int64_t acc = horner(cq, chosen_frac, f);
            int code = static_cast<int>((acc - ONE + HALF) >> ROUND_SHIFT);

            int out_exp = -int_part - 1;
            if (code == 128) { code = 0; ++out_exp; }
            const uint16_t got =
                static_cast<uint16_t>(((127 + out_exp) << 7) | code);

            if (got != ref_code(exp, mant)) {
                if (++mismatches <= 10)
                    std::cerr << "Self-check mismatch exp=" << exp
                              << " mant=" << mant << '\n';
            }
        }
    }
    if (mismatches) {
        std::cerr << "\nSelf-check FAILED with " << mismatches
                  << " mismatches -- nothing written.\n";
        return 1;
    }
    std::cout << "\nself-check     : 0 mismatches over " << total_ladder
              << " ladder inputs\n";

    const int payload_bits = chosen_w * NCOEF + static_cast<int>(tail.size()) * 7;
    write_cpp_header(chosen_w, chosen_frac, cq, tail, tail_lo, tail_hi, zero_lo,
                     payload_bits, chosen_slack);
    write_sv_package(chosen_w, chosen_frac, cq, tail, tail_lo, tail_hi, zero_lo,
                     payload_bits);

    std::cout << "payload        : " << payload_bits << " bits ("
              << (chosen_w * NCOEF) << " coefficients + "
              << (tail.size() * 7) << " tail)\n"
              << "DSP slices     : " << DEG << " Horner + 1 constant multiply\n\n"
              << "Wrote src/approximations/bf16_expe_poly4_table.hpp\n"
              << "Wrote src/rtl/bf16_expe_poly4_rom.sv\n";
    return 0;
}

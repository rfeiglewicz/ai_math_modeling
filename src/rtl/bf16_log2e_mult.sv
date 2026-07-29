// =============================================================================
// bf16_log2e_mult.sv
// Multiplies mantissa source (1.7 format) by log2(e) constant (1.22 format).
// Used only when base2=0 (exp(x) mode). When base2=1, output equals input.
//
// After multiplication, an optional RNE (Round to Nearest Even) rounding
// reduces the 2.29 product to 2.MANT_MULT_ROUND_FRAC precision.
// Default MANT_MULT_ROUND_FRAC = MANT_MULT_F (29) = full precision (no bits dropped).
//
// Parameterizable:
//   LOG2E_I               - integer bits of the constant (default: 1)
//   LOG2E_F               - fractional bits of the constant (default: 22)
//   LOG2E_VAL             - fixed-point value of log2(e) (default: 0x5c551d)
//   MANT_MULT_ROUND_FRAC  - fractional bits to keep after RNE (default: 29)
//
// NOTE: The output port mant_out provides the effective mantissa (either
// mant_src for base2 mode, or mant_mult_rne for exp(x) mode). Both are provided
// so the top level does not require a MUX chain.
// =============================================================================

module bf16_log2e_mult
    import bf16_exp2_pkg::*;
#(
    parameter int LOG2E_I   = LOG2E_I_DEFAULT,    // 1
    parameter int LOG2E_F   = LOG2E_F_DEFAULT,    // 22
    parameter int LOG2E_VAL = LOG2E_VAL_DEFAULT,  // 0x5c551d
    parameter int MANT_MULT_ROUND_FRAC = MANT_MULT_F,  // 29 = no rounding
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Retiming:
    //   1 inserts a register between the multiply and the RNE rounding adder.
    //   2 additionally registers the rounded/muxed result.
    //
    // Level 2 exists because REGISTER_OUTPUT alone is not enough: the consumer
    // (bf16_unified_shift) starts with a DSP multiply, so Vivado absorbs the
    // output register into that DSP's BREG and the RNE adder ends up sharing a
    // stage with the multiply Tco on one side and the DSP setup on the other.
    // With two registers here Vivado can absorb one into BREG and still keep a
    // fabric register directly behind the adder.
    //
    // Bit-exact; each level costs one extra cycle of latency.
    parameter int EXTRA_STAGES    = 0
)(
    input  logic                                   clk,
    input  logic                                   rst_n,
    input  logic                                   pipe_en,
    input  logic                                   base2,
    input  logic [MANT_SRC_W-1:0]                  mant_src,  // 1.7 unsigned
    // Multiplication result: 2.29 = MANT_MULT_W bits
    output logic [MANT_MULT_W-1:0]                 mant_out   // Mux output
);

    localparam int LOG2E_W = LOG2E_I + LOG2E_F;

    // -------------------------------------------------------------------------
    // RNE rounding parameters
    // -------------------------------------------------------------------------
    // Bits to discard from the 2.29 product
    localparam int DISCARD_BITS = MANT_MULT_F - MANT_MULT_ROUND_FRAC;
    // Rounded width = MANT_MULT_I + MANT_MULT_ROUND_FRAC
    localparam int ROUNDED_W = MANT_MULT_I + MANT_MULT_ROUND_FRAC;

    logic [MANT_MULT_W-1:0] mant_mult_raw;
    logic [MANT_MULT_W-1:0] mant_mult;
    logic [MANT_SRC_W-1:0]  mant_src_d;
    logic                   base2_d;
    logic [MANT_MULT_W-1:0] mant_mult_rne;
    logic [MANT_MULT_W-1:0] mant_out_raw;
    logic [MANT_MULT_W-1:0] mant_out_comb;

    // Product: mant_src (1.7) * log2e (1.22) = 2.29 = 31 bits
    assign mant_mult_raw = mant_src * LOG2E_VAL[LOG2E_W-1:0];

    // -------------------------------------------------------------------------
    // Retiming register: separates the multiply from the RNE rounding adder.
    // Without it the stage is DSP multiply + 5-deep carry chain + mux, and the
    // output register gets absorbed into the NEXT block's DSP input register,
    // so the whole thing counts as one stage. mant_src and base2 ride along
    // because the base2 bypass has to stay aligned with the rounded path.
    // The register lands in the DSP48 PREG, so it costs no fabric.
    // -------------------------------------------------------------------------
    generate
        if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_mult_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mant_mult <= '0; mant_src_d <= '0; base2_d <= 1'b1;
                end else if (pipe_en) begin
                    mant_mult <= mant_mult_raw; mant_src_d <= mant_src; base2_d <= base2;
                end
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_mult_reg
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    mant_mult <= mant_mult_raw; mant_src_d <= mant_src; base2_d <= base2;
                end
            end
        end else begin : gen_mult_wire
            assign mant_mult  = mant_mult_raw;
            assign mant_src_d = mant_src;
            assign base2_d    = base2;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // RNE (Round to Nearest Even) rounding of mant_mult from 2.29 to
    // 2.MANT_MULT_ROUND_FRAC.
    //
    // When MANT_MULT_ROUND_FRAC == MANT_MULT_F (29), DISCARD_BITS=0 and
    // the logic is optimized away (no rounding).
    // -------------------------------------------------------------------------
    generate
        if (DISCARD_BITS == 0) begin : gen_no_round
            // No rounding needed - full precision
            assign mant_mult_rne = mant_mult;
        end else begin : gen_rne
            logic                       lsb_bit;    // bit at DISCARD_BITS
            logic                       guard_bit;  // bit at DISCARD_BITS-1
            logic                       sticky_bit; // OR of bits below guard
            logic                       round_up;
            logic [ROUNDED_W-1:0]       truncated;
            logic [ROUNDED_W-1:0]       rounded;

            assign lsb_bit   = mant_mult[DISCARD_BITS];
            assign guard_bit  = mant_mult[DISCARD_BITS-1];

            if (DISCARD_BITS > 1) begin : gen_sticky
                assign sticky_bit = |mant_mult[DISCARD_BITS-2:0];
            end else begin : gen_no_sticky
                assign sticky_bit = 1'b0;
            end

            // RNE: round up if guard=1 AND (lsb=1 OR sticky=1)
            assign round_up  = guard_bit & (lsb_bit | sticky_bit);
            assign truncated = mant_mult[MANT_MULT_W-1 -: ROUNDED_W];
            assign rounded   = truncated + {{(ROUNDED_W-1){1'b0}}, round_up};

            // Zero-extend rounded value back to MANT_MULT_W width
            assign mant_mult_rne = {rounded, {DISCARD_BITS{1'b0}}};
        end
    endgenerate

    // Mux: select base2 or base-e path
    // For base2 mode, zero-extend mant_src to MANT_MULT_W bits by left-shifting
    // to align in the 2.29 format (mant_src is 1.7, MANT_MULT_W is 31 = 2.29)
    // So mant_src (1.7) needs MANT_MULT_F - MANT_SRC_F = 29 - 7 = 22 extra bits
    localparam int ALIGN = MANT_MULT_F - MANT_SRC_F;  // = 22

    always_comb begin
        if (base2_d)
            mant_out_raw = MANT_MULT_W'({mant_src_d, {ALIGN{1'b0}}});
        else
            mant_out_raw = mant_mult_rne;
    end

    // -------------------------------------------------------------------------
    // Second retiming register: isolates the RNE adder + base2 mux so that they
    // no longer sit between two DSP blocks.
    //
    // dont_touch is required. Without it Vivado packs this register (and the
    // REGISTER_OUTPUT one) into the consumer DSP's two-deep B input register,
    // which leaves the adder still hanging directly off the multiplier's P
    // output: 2.29 ns of DSP Tco plus the whole carry chain in one stage.
    // Forcing it into fabric costs MANT_MULT_W flip-flops and buys a clean cut.
    // -------------------------------------------------------------------------
    generate
        if (EXTRA_STAGES >= 2) begin : gen_rne_reg
            (* dont_touch = "true" *) logic [MANT_MULT_W-1:0] mant_rne_q;
            if (RESET_DATAPATH) begin : gen_rst
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)       mant_rne_q <= '0;
                    else if (pipe_en) mant_rne_q <= mant_out_raw;
                end
            end else begin : gen_nrst
                always_ff @(posedge clk) begin
                    if (pipe_en) mant_rne_q <= mant_out_raw;
                end
            end
            assign mant_out_comb = mant_rne_q;
        end else begin : gen_rne_wire
            assign mant_out_comb = mant_out_raw;
        end
    endgenerate

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      mant_out <= '0;
                else if (pipe_en) mant_out <= mant_out_comb;
            end
        end else if (REGISTER_OUTPUT) begin : gen_reg_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) mant_out <= mant_out_comb;
            end
        end else begin : gen_comb
            assign mant_out = mant_out_comb;
        end
    endgenerate

endmodule : bf16_log2e_mult

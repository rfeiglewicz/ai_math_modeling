// =============================================================================
// bf16_exp2_optim_log2e_mult.sv
// Multiplies the mantissa source (Q1.7) by log2(e) (Q1.22) and rounds the
// Q2.29 product straight down to Q2.21. Used only when base2=0 (e^x mode);
// when base2=1 the mantissa is passed through, left-aligned to the same
// Q2.21 field.
//
// Difference from bf16_log2e_mult: the wide product never leaves this module.
// Production keeps all 31 bits of Q2.29 on the wire and only rounds if asked
// to; here the rounded Q2.21 value IS the interface, which is what shrinks the
// unified shift register, the polynomial input and everything downstream.
//
// 21 fractional bits is the RNE minimum. 20 breaks expe. Truncation would need
// 22, which this module cannot produce because MANT_MULT_F is its own output
// width -- that is the reason it rounds rather than truncates.
//
// ROUND_MODE selects between RNE and round-half-up. They differ only on an
// exact tie, and with LOG2E_VAL = 0x5c551d there is exactly ONE tie in the
// whole input space: mant_src = 0x80, where the discarded 8 bits are 0x80 and
// the lsb is 0, so RNE goes down and half-up goes up. That is the mantissa
// field being all zeros, i.e. inputs of the form +/-2^E, and only on the
// base-e path -- base2 bypasses this rounder entirely. The resulting 1 ULP at
// 2^-21 does not survive the final rounding to 8 mantissa bits: both modes are
// bit-exact against bf16_exp2_approx over all 65536 patterns x both bases
// (make rtl_optim_verify_pipelined, make rtl_optim_half_up).
//
// That equality is an empirical property of this table and these widths, not
// an identity -- which is why half-up has its own exhaustive run instead of
// inheriting the RNE result. Note the width sweep fails at 20 bits for both
// modes, so the margin carrying this is roughly one bit, not a comfortable one.
//
// The choice is therefore free at the output and worth making deliberately at
// the timing level: with the retiming maxed out, the RNE adder in this module
// IS the critical path of the whole core. Measured on xc7a200t (make sweep_optim):
//
//   ROUND_MODE=0 RNE      164.8 MHz, 8 levels, critical path starts here
//   ROUND_MODE=1 half-up  177.5 MHz, 5 levels, critical path moves to the rounder
//
// for the same 368/369 LUT. The difference is the sticky OR and the lsb term
// feeding the adder's carry-in; half-up needs only the guard bit.
//
// Parameterizable:
//   LOG2E_I/F/VAL         - the constant, Q1.22 by default
//   MANT_MULT_ROUND_FRAC  - fractional bits kept after rounding (default 21).
//                           Lower values leave (21 - value) zero LSBs in the
//                           Q2.21 field, exactly as production does, so the
//                           minimum can be re-derived from the RTL itself.
//   ROUND_MODE            - 0 = RNE, 1 = round-half-up, 2 = truncate
// =============================================================================

module bf16_exp2_optim_log2e_mult
    import bf16_exp2_optim_pkg::*;
#(
    parameter int LOG2E_I   = LOG2E_I_DEFAULT,    // 1
    parameter int LOG2E_F   = LOG2E_F_DEFAULT,    // 22
    parameter int LOG2E_VAL = LOG2E_VAL_DEFAULT,  // 0x5c551d
    parameter int MANT_MULT_ROUND_FRAC = MANT_MULT_F,  // 21
    parameter int ROUND_MODE      = 0,            // 0=RNE, 1=half-up, 2=trunc
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Retiming:
    //   1 inserts a register between the multiply and the rounding adder.
    //   2 additionally registers the rounded/muxed result.
    // Bit-exact; each level costs one extra cycle of latency.
    parameter int EXTRA_STAGES    = 0
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    pipe_en,
    input  logic                    base2,
    input  logic [MANT_SRC_W-1:0]   mant_src,   // Q1.7 unsigned
    output logic [MANT_MULT_W-1:0]  mant_out    // Q2.21 unsigned
);

    localparam int LOG2E_W = LOG2E_I + LOG2E_F;

    // Bits discarded from the Q2.29 raw product. 8 by default.
    localparam int DISCARD_BITS = MANT_MULT_FULL_F - MANT_MULT_ROUND_FRAC;
    localparam int ROUNDED_W    = MANT_MULT_I + MANT_MULT_ROUND_FRAC;
    // Zero LSBs left in the Q2.21 field when rounding harder than the default.
    localparam int ZERO_LSBS    = MANT_MULT_F - MANT_MULT_ROUND_FRAC;

    // The output field is Q2.MANT_MULT_F, so asking to keep MORE fractional
    // bits than that has nowhere to put them. Anything above 21 would have to
    // widen MANT_MULT_F, and with it the unified shift register and the whole
    // downstream datapath -- which is the opposite of the point. In particular
    // the truncation minimum of 22 bits is not reachable from here; use RNE or
    // round-half-up, both of which reach 21.
    if (MANT_MULT_ROUND_FRAC > MANT_MULT_F) begin : gen_bad_round_frac
        $error("MANT_MULT_ROUND_FRAC=%0d exceeds MANT_MULT_F=%0d; widen the package first",
               MANT_MULT_ROUND_FRAC, MANT_MULT_F);
    end

    // Rounding up can never carry out of ROUNDED_W: the largest product is
    // 255 * 0x5c551d = 0x5BF5_C1E3, and 2.99 < 4 leaves the Q2.x integer field
    // with room to spare.
    logic [MANT_MULT_FULL_W-1:0] mant_mult_raw;
    logic [MANT_MULT_FULL_W-1:0] mant_mult;
    logic [MANT_SRC_W-1:0]       mant_src_d;
    logic                        base2_d;
    logic [ROUNDED_W-1:0]        rounded;
    logic [MANT_MULT_W-1:0]      mant_out_raw;
    logic [MANT_MULT_W-1:0]      mant_out_comb;

    // Q1.7 * Q1.22 = Q2.29, 31 bits. One DSP48E1, no cascade.
    assign mant_mult_raw = mant_src * LOG2E_VAL[LOG2E_W-1:0];

    // -------------------------------------------------------------------------
    // Retiming register: separates the multiply from the rounding adder so the
    // register lands in the DSP48 PREG instead of the consumer's input register.
    // mant_src and base2 ride along because the base2 bypass has to stay
    // aligned with the rounded path.
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
    // Round Q2.29 -> Q2.MANT_MULT_ROUND_FRAC
    // -------------------------------------------------------------------------
    generate
        if (DISCARD_BITS == 0) begin : gen_no_round
            assign rounded = mant_mult[MANT_MULT_FULL_W-1 -: ROUNDED_W];
        end else begin : gen_round
            logic                 lsb_bit;
            logic                 guard_bit;
            logic                 sticky_bit;
            logic                 round_up;
            logic [ROUNDED_W-1:0] truncated;

            assign lsb_bit   = mant_mult[DISCARD_BITS];
            assign guard_bit = mant_mult[DISCARD_BITS-1];

            if (DISCARD_BITS > 1) begin : gen_sticky
                assign sticky_bit = |mant_mult[DISCARD_BITS-2:0];
            end else begin : gen_no_sticky
                assign sticky_bit = 1'b0;
            end

            // RNE      : up if guard and (lsb or sticky)  -- ties to even
            // half-up  : up if guard                      -- one constant add
            // truncate : never
            if (ROUND_MODE == 1) begin : gen_half_up
                assign round_up = guard_bit;
            end else if (ROUND_MODE == 2) begin : gen_trunc
                assign round_up = 1'b0;
            end else begin : gen_rne
                assign round_up = guard_bit & (lsb_bit | sticky_bit);
            end

            assign truncated = mant_mult[MANT_MULT_FULL_W-1 -: ROUNDED_W];
            assign rounded   = truncated + {{(ROUNDED_W-1){1'b0}}, round_up};
        end
    endgenerate

    // -------------------------------------------------------------------------
    // base2 / base-e mux.
    //
    // Q1.7 sits in the Q2.21 field with MANT_MULT_F - MANT_SRC_F = 14 zero
    // LSBs and one spare integer bit, so the bypass is pure wiring.
    // -------------------------------------------------------------------------
    localparam int ALIGN = MANT_MULT_F - MANT_SRC_F;  // 14

    always_comb begin
        if (base2_d)
            mant_out_raw = MANT_MULT_W'({mant_src_d, {ALIGN{1'b0}}});
        else
            mant_out_raw = MANT_MULT_W'({rounded, {ZERO_LSBS{1'b0}}});
    end

    // -------------------------------------------------------------------------
    // Second retiming register: isolates the rounding adder and the mux so
    // they do not sit between two DSP blocks. dont_touch stops Vivado packing
    // it back into the consumer's input register, which would put the adder
    // right behind the multiplier's P output again.
    // -------------------------------------------------------------------------
    generate
        if (EXTRA_STAGES >= 2) begin : gen_rnd_reg
            (* dont_touch = "true" *) logic [MANT_MULT_W-1:0] mant_rnd_q;
            if (RESET_DATAPATH) begin : gen_rst
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)       mant_rnd_q <= '0;
                    else if (pipe_en) mant_rnd_q <= mant_out_raw;
                end
            end else begin : gen_nrst
                always_ff @(posedge clk) begin
                    if (pipe_en) mant_rnd_q <= mant_out_raw;
                end
            end
            assign mant_out_comb = mant_rnd_q;
        end else begin : gen_rnd_wire
            assign mant_out_comb = mant_out_raw;
        end
    endgenerate

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       mant_out <= '0;
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

endmodule : bf16_exp2_optim_log2e_mult

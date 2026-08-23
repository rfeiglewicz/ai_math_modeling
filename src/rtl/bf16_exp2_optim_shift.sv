// =============================================================================
// bf16_exp2_optim_shift.sv
// Applies the input exponent to the Q2.21 mantissa and splits the result into
// the polynomial input and the exponent bias.
//
//   unified = mant_in << exponent    (exponent >= 0)
//   unified = mant_in >> -exponent   (exponent <  0)
//
//   x             = unified[UNIFIED_F-1 -: X_F]   fractional part, truncated
//   exponent_bias = -unified[UNIFIED_W-1 : UNIFIED_F]
//
// Two differences from bf16_unified_shift, both consequences of UNIFIED_F
// being equal to MANT_MULT_F rather than 9 bits wider:
//
//   1. There is no zero padding below the mantissa, so a right shift really
//      does drop bits. That is intended: the C++ model truncates identically
//      (ac_fixed defaults to AC_TRN), and the exhaustive study confirms the
//      dropped bits never change the BF16 result.
//
//   2. The one-hot DSP shifter is not available. Production can collapse both
//      shift directions into a single left shift only because the 9 pad bits
//      guarantee the right shift is lossless; here they do not exist. The
//      fabric barrel shifter is 30 bits wide instead of 47, so it is roughly
//      a third of the cost it was, and moving it to a DSP would have to
//      re-introduce the padding it was the point of removing.
//
// Bit widths:
//   mant_in : MANT_MULT_W = 23  (Q2.21 unsigned)
//   unified : UNIFIED_W   = 30  (Q9.21 unsigned)
//   x       : X_W         = 17  (Q0.17 unsigned)
//   int_part: IN_CONV_INT_W = 9 (signed, negated)
// =============================================================================

module bf16_exp2_optim_shift
    import bf16_exp2_optim_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Retiming: 1 registers the shifted value before the slice logic.
    // Bit-exact; costs one cycle of latency.
    parameter int EXTRA_STAGES    = 0
)(
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            pipe_en,
    input  logic [MANT_MULT_W-1:0]          mant_in,    // Q2.21 unsigned
    input  logic signed [8:0]               exponent,   // unbiased input exponent
    output logic [X_W-1:0]                  x,          // Q0.17 polynomial input
    output logic signed [IN_CONV_INT_W-1:0] int_part    // negated integer part
);

    // FRAC_PAD is zero by construction: UNIFIED_F == MANT_MULT_F.
    localparam int FRAC_PAD = UNIFIED_F - MANT_MULT_F;

    logic [UNIFIED_W-1:0] mant_aligned;
    logic [UNIFIED_W-1:0] shifted_comb;
    logic [UNIFIED_W-1:0] unified_shifted;

    assign mant_aligned = UNIFIED_W'({mant_in, {FRAC_PAD{1'b0}}});

    // Exponent range is [-9, +7]; anything outside is masked by bf16_early_out
    // at the top level, so out-of-range shift amounts are don't-care.
    always_comb begin
        if ($signed(exponent) >= 0)
            shifted_comb = mant_aligned << exponent;
        else
            shifted_comb = mant_aligned >> (-exponent);
    end

    generate
        if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_shift_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       unified_shifted <= '0;
                else if (pipe_en) unified_shifted <= shifted_comb;
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_shift_reg
            always_ff @(posedge clk) begin
                if (pipe_en) unified_shifted <= shifted_comb;
            end
        end else begin : gen_shift_wire
            assign unified_shifted = shifted_comb;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Split.
    //
    // Only the top X_F of the UNIFIED_F fraction bits reach the multiplier;
    // the rest are dropped, which is what makes the polynomial input 17 bits
    // instead of 21.
    //
    // int_part is a 9-bit signed negation of a 9-bit unsigned field, so the
    // base-e path can wrap: the largest value the front end can produce is
    // 2.8738 * 2^7 = 367, and -367 does not fit. This matches production
    // exactly and is harmless -- the wrapped exponent is large and positive,
    // bf16_recompose adds the 127 bias, that wraps in turn to a negative
    // 9-bit value, and the result is flushed to +0.0. Which is the right
    // answer: every input that reaches here is e^-178 or smaller.
    // -------------------------------------------------------------------------
    logic [X_W-1:0]                  x_comb;
    logic signed [IN_CONV_INT_W-1:0] int_part_comb;

    assign x_comb        = unified_shifted[UNIFIED_F-1 -: X_F];
    assign int_part_comb = -signed'(unified_shifted[UNIFIED_W-1 : UNIFIED_F]);

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    x        <= '0;
                    int_part <= '0;
                end else if (pipe_en) begin
                    x        <= x_comb;
                    int_part <= int_part_comb;
                end
            end
        end else if (REGISTER_OUTPUT) begin : gen_reg_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    x        <= x_comb;
                    int_part <= int_part_comb;
                end
            end
        end else begin : gen_comb
            assign x        = x_comb;
            assign int_part = int_part_comb;
        end
    endgenerate

endmodule : bf16_exp2_optim_shift

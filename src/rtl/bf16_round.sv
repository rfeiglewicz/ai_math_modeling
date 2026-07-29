// =============================================================================
// bf16_round.sv
// Rounds the polynomial result to BF16 precision using Round-to-Nearest-Even.
//
// Steps (matches C++ bf16_exp2_core_approx post-poly section):
//   1. Combine poly_exponent + exponent_bias -> final_exponent
//   2. Check subnormal (final_exponent < BF16_MIN_EXP)
//   3. Compute shift_val (bits to discard): BASE_SHIFT + subnormal_extra
//   4. Extract LSB, Guard, Sticky bits
//   5. round_up = guard && (lsb || sticky)
//   6. Right-shift poly_mantissa by shift_val, increment if round_up
//   7. Handle carry-out: if overflow bit set, shift right and adjust exponent
//   8. Build fp_raw_t output (normal / denormal / zero)
//
// Bit widths:
//   poly_mantissa : POLY_OUT_W = 59 bits
//   exponent_bias : signed [IN_CONV_INT_W-1:0] = 9 bits (from int_part)
//   result_m_ext  : EXT_MANT_W = 9 bits (carry + hidden + 7 mantissa)
// =============================================================================

module bf16_round
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Retiming: the rounder is one long serial dependency --
    //   exponent add -> subnormal compare -> shift amount -> indexed bit
    //   extract -> sticky mask + reduce -> variable right shift ->
    //   increment -> carry fixup -> assemble
    // which measures 19 logic levels as a single stage. EXTRA_STAGES cuts it:
    //   1 - after the shift amount is known (exponent arithmetic split off)
    //   2 - additionally after the shift + round increment, leaving only the
    //       carry fixup and field assembly in the last stage
    // Bit-exact; costs EXTRA_STAGES cycles of latency.
    parameter int EXTRA_STAGES    = 0
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                pipe_en,
    input  logic [POLY_OUT_W-1:0]               poly_mantissa,
    input  logic signed [8:0]                   poly_exponent,
    input  logic signed [IN_CONV_INT_W-1:0]     exponent_bias,  // int_part
    output fp_raw_t                             rounded_fp
);

    localparam int CARRY_BIT_IDX  = EXT_MANT_W - 1;
    localparam int HIDDEN_BIT_IDX = BF16_MANT_BITS;

    // -------------------------------------------------------------------------
    // Block 1: exponent arithmetic -> how many bits get discarded
    // -------------------------------------------------------------------------
    logic signed [8:0] final_exponent;
    logic              is_sub;
    logic signed [8:0] shift_9;

    always_comb begin
        final_exponent = poly_exponent + 9'(signed'(exponent_bias));
        is_sub         = ($signed(final_exponent) < $signed(9'(BF16_MIN_EXP)));
        if (is_sub)
            shift_9 = 9'(signed'(BASE_SHIFT)) + (9'(signed'(BF16_MIN_EXP)) - final_exponent);
        else
            shift_9 = 9'(signed'(BASE_SHIFT));
    end

    // --- retiming register 1 ---
    logic signed [8:0]        p1_final_exp;
    logic                     p1_is_sub;
    logic signed [8:0]        p1_shift;
    logic [POLY_OUT_W-1:0]    p1_mant;

    generate
        if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_p1_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    p1_final_exp <= '0; p1_is_sub <= 1'b0;
                    p1_shift     <= '0; p1_mant   <= '0;
                end else if (pipe_en) begin
                    p1_final_exp <= final_exponent; p1_is_sub <= is_sub;
                    p1_shift     <= shift_9;        p1_mant   <= poly_mantissa;
                end
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_p1
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    p1_final_exp <= final_exponent; p1_is_sub <= is_sub;
                    p1_shift     <= shift_9;        p1_mant   <= poly_mantissa;
                end
            end
        end else begin : gen_p1_wire
            assign p1_final_exp = final_exponent;
            assign p1_is_sub    = is_sub;
            assign p1_shift     = shift_9;
            assign p1_mant      = poly_mantissa;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Block 2: extract L/G/S, shift the mantissa down, apply the RNE increment
    // -------------------------------------------------------------------------
    logic                  lsb_bit;
    logic                  guard_bit;
    logic                  sticky_bit;
    logic                  round_up;
    logic [EXT_MANT_W-1:0] sum_m_ext;
    logic signed [8:0]     base_exp;

    // Sticky is the OR of every bit below the guard bit.
    //
    // Two traps here, both of which cost a long carry chain if written the
    // obvious way:
    //   1. the mask (1 << (shift-1)) - 1 becomes a 59-bit subtractor, i.e. a
    //      15-deep borrow chain, even though it is only a thermometer code;
    //   2. |(mant & mask) becomes a 14-deep CARRY4 comparison instead of a
    //      LUT tree.
    // Both are rewritten below. The mask is a per-bit compare
    //      mask[i] = shift > i+1
    // which is one LUT level and, as a bonus, is uniformly correct for the two
    // edge cases the original spelled out separately: shift <= 1 gives an all
    // zero mask, and shift >= POLY_OUT_W gives an all ones mask. The reduction
    // is then grouped so it maps to a two-level tree.
    localparam int STICKY_GRP    = 8;
    localparam int STICKY_NGRP   = (POLY_OUT_W + STICKY_GRP - 1) / STICKY_GRP;
    localparam int STICKY_PAD_W  = STICKY_NGRP * STICKY_GRP;

    logic [POLY_OUT_W-1:0]   sticky_mask;
    logic [POLY_OUT_W-1:0]   sticky_masked;
    logic [STICKY_PAD_W-1:0] sticky_padded;
    logic [STICKY_NGRP-1:0]  sticky_part;

    generate
        for (genvar i = 0; i < POLY_OUT_W; i++) begin : gen_sticky_mask
            assign sticky_mask[i] = ($signed(p1_shift) > $signed(9'(i + 1)));
        end
    endgenerate

    assign sticky_masked = p1_mant & sticky_mask;
    assign sticky_padded = STICKY_PAD_W'(sticky_masked);

    generate
        for (genvar g = 0; g < STICKY_NGRP; g++) begin : gen_sticky_grp
            assign sticky_part[g] = |sticky_padded[g*STICKY_GRP +: STICKY_GRP];
        end
    endgenerate

    assign sticky_bit = |sticky_part;

    always_comb begin
        lsb_bit   = 1'b0;
        guard_bit = 1'b0;
        round_up  = 1'b0;
        sum_m_ext = '0;

        lsb_bit   = (p1_shift < POLY_OUT_W) ? p1_mant[p1_shift] : 1'b0;
        guard_bit = ((p1_shift > 0) && (p1_shift <= POLY_OUT_W)) ? p1_mant[p1_shift-1] : 1'b0;

        round_up  = guard_bit & (lsb_bit | sticky_bit);

        sum_m_ext = (p1_shift < POLY_OUT_W) ? EXT_MANT_W'(p1_mant >> p1_shift) : '0;
        if (round_up) sum_m_ext = sum_m_ext + 1'b1;

        base_exp = p1_is_sub ? 9'(signed'(BF16_MIN_EXP)) : p1_final_exp;
    end

    // --- retiming register 2 ---
    logic [EXT_MANT_W-1:0] p2_m_ext;
    logic signed [8:0]     p2_base_exp;
    logic                  p2_is_sub;

    generate
        if (EXTRA_STAGES >= 2 && RESET_DATAPATH) begin : gen_p2_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    p2_m_ext <= '0; p2_base_exp <= '0; p2_is_sub <= 1'b0;
                end else if (pipe_en) begin
                    p2_m_ext <= sum_m_ext; p2_base_exp <= base_exp; p2_is_sub <= p1_is_sub;
                end
            end
        end else if (EXTRA_STAGES >= 2) begin : gen_p2
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    p2_m_ext <= sum_m_ext; p2_base_exp <= base_exp; p2_is_sub <= p1_is_sub;
                end
            end
        end else begin : gen_p2_wire
            assign p2_m_ext    = sum_m_ext;
            assign p2_base_exp = base_exp;
            assign p2_is_sub   = p1_is_sub;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Block 3: post-round carry fixup and BF16 field assembly
    // -------------------------------------------------------------------------
    logic [EXT_MANT_W-1:0] result_m_ext;
    logic signed [8:0]     adjusted_exp;
    fp_raw_t               rounded_comb;

    always_comb begin
        result_m_ext = p2_m_ext;
        adjusted_exp = p2_base_exp;
        if (result_m_ext[CARRY_BIT_IDX]) begin
            adjusted_exp = adjusted_exp + 9'sd1;
            result_m_ext = result_m_ext >> 1;
        end

        rounded_comb      = '0;
        rounded_comb.sign = 1'b0;
        if (result_m_ext == 0) begin
            rounded_comb.status.is_zero = 1'b1;
        end else if (p2_is_sub && !result_m_ext[HIDDEN_BIT_IDX]) begin
            rounded_comb.mantissa           = result_m_ext[BF16_MANT_BITS-1 : 0];
            rounded_comb.hidden_bit         = 1'b0;
            rounded_comb.exponent           = 9'(signed'(BF16_MIN_EXP - 1));
            rounded_comb.status.is_denormal = 1'b1;
        end else begin
            rounded_comb.mantissa   = result_m_ext[BF16_MANT_BITS-1 : 0];
            rounded_comb.hidden_bit = 1'b1;
            rounded_comb.exponent   = adjusted_exp;
        end
    end

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      rounded_fp <= '0;
                else if (pipe_en) rounded_fp <= rounded_comb;
            end
        end else if (REGISTER_OUTPUT) begin : gen_reg_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) rounded_fp <= rounded_comb;
            end
        end else begin : gen_comb
            assign rounded_fp = rounded_comb;
        end
    endgenerate

endmodule : bf16_round

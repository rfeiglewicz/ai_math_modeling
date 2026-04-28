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
    parameter bit REGISTER_OUTPUT = 1'b0
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
    // Intermediate signals
    // -------------------------------------------------------------------------
    logic signed [8:0]       final_exponent;
    logic                    is_sub;
    logic signed [8:0]       shift_9;  // shift_val as signed 9-bit (enough range)

    logic                    lsb_bit;
    logic                    guard_bit;
    logic                    sticky_bit;
    logic                    round_up;

    logic [EXT_MANT_W-1:0]  result_m_ext;
    logic signed [8:0]       adjusted_exp;
    fp_raw_t                 rounded_comb;

    always_comb begin
        // Default initializations to avoid latches
        final_exponent = '0;
        is_sub         = 1'b0;
        shift_9        = '0;
        lsb_bit        = 1'b0;
        guard_bit      = 1'b0;
        sticky_bit     = 1'b0;
        round_up       = 1'b0;
        result_m_ext   = '0;
        adjusted_exp   = '0;
        rounded_comb   = '0;

        // Step 1: Combine exponents
        final_exponent = poly_exponent + 9'(signed'(exponent_bias));

        // Step 2: Subnormal check
        is_sub = ($signed(final_exponent) < $signed(9'(BF16_MIN_EXP)));

        // Step 3: shift_val
        if (is_sub)
            shift_9 = 9'(signed'(BASE_SHIFT)) + (9'(signed'(BF16_MIN_EXP)) - final_exponent);
        else
            shift_9 = 9'(signed'(BASE_SHIFT));

        // Step 4: Extract rounding bits
        lsb_bit    = (shift_9 < POLY_OUT_W) ? poly_mantissa[shift_9]     : 1'b0;
        guard_bit  = ((shift_9 > 0) && (shift_9 <= POLY_OUT_W)) ? poly_mantissa[shift_9-1] : 1'b0;

        // Sticky: OR of all bits below guard
        if (shift_9 > 1) begin
            if (shift_9 >= POLY_OUT_W) begin
                sticky_bit = |poly_mantissa;
            end else begin
                // bits [0 .. shift_9-2]
                logic [POLY_OUT_W-1:0] sticky_mask;
                sticky_mask = (POLY_OUT_W'(1'b1) << (shift_9 - 1)) - 1;
                sticky_bit  = |(poly_mantissa & sticky_mask);
            end
        end else begin
            sticky_bit = 1'b0;
        end

        // Step 5: RNE decision
        round_up = guard_bit & (lsb_bit | sticky_bit);

        // Step 6: Shift and round
        result_m_ext = (shift_9 < POLY_OUT_W) ?
                       EXT_MANT_W'(poly_mantissa >> shift_9) :
                       '0;
        if (round_up)
            result_m_ext = result_m_ext + 1'b1;

        // Step 7: Post-round carry
        adjusted_exp = is_sub ? BF16_MIN_EXP : final_exponent;
        if (result_m_ext[CARRY_BIT_IDX]) begin
            adjusted_exp = adjusted_exp + 9'sd1;
            result_m_ext = result_m_ext >> 1;
        end

        // Step 8: Build result
        rounded_comb = '0;
        rounded_comb.sign = 1'b0;
        if (result_m_ext == 0) begin
            rounded_comb.status.is_zero = 1'b1;
        end else if (is_sub && !result_m_ext[HIDDEN_BIT_IDX]) begin
            // Denormal output
            rounded_comb.mantissa           = result_m_ext[BF16_MANT_BITS-1 : 0];
            rounded_comb.hidden_bit         = 1'b0;
            rounded_comb.exponent           = 9'(signed'(BF16_MIN_EXP - 1));
            rounded_comb.status.is_denormal = 1'b1;
        end else begin
            // Normal output
            rounded_comb.mantissa   = result_m_ext[BF16_MANT_BITS-1 : 0];
            rounded_comb.hidden_bit = 1'b1;
            rounded_comb.exponent   = adjusted_exp;
        end
    end

    generate
        if (REGISTER_OUTPUT) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      rounded_fp <= '0;
                else if (pipe_en) rounded_fp <= rounded_comb;
            end
        end else begin : gen_comb
            assign rounded_fp = rounded_comb;
        end
    endgenerate

endmodule : bf16_round

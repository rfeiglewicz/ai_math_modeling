// =============================================================================
// bf16_unified_shift.sv
// Converts the mantissa fixed-point value (from log2e mux output) into the
// unified representation and applies the input exponent shift.
//
// Computes:
//   unified_val = mant_in << exponent   (if exponent >= 0)
//   unified_val = mant_in >> (-exponent) (if exponent < 0)
//
// Then separates the result into:
//   frac_part = lowest IN_F bits of unified_val (fractional part -> polynomial x)
//   int_part  = highest IN_CONV_INT_W bits -> -exponent_bias
//
// C++ model equivalent:
//   val = (unified_t)mant_val;
//   val <<= (or >>= ) exponent;
//   mant_val = val.slc<IN_F>(0);
//   exponent_bias = -(int)val.to_int();
//
// Bit widths (defaults match C++ namespace):
//   mant_in : MANT_MULT_W = 31 bits (2.29 format)
//   unified  : IN_CONV_INT_W + IN_F = 9 + 38 = 47 bits
//   frac_part: IN_F = 38 bits
//   int_part : IN_CONV_INT_W = 9 bits (signed, negated)
// =============================================================================

module bf16_unified_shift
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0
)(
    input  logic                                           clk,
    input  logic                                           rst_n,
    input  logic [MANT_MULT_W-1:0]                         mant_in,    // 2.29 unsigned from log2e_mult
    input  logic signed [8:0]                              exponent,   // Unbiased input exponent
    output logic [IN_F-1:0]                                frac_part,  // 38-bit fractional (poly x)
    output logic signed [IN_CONV_INT_W-1:0]                int_part    // 9-bit integer (negated bias)
);

    // -------------------------------------------------------------------------
    // Place mant_in into unified 47-bit register at the correct position.
    // The unified format is (IN_CONV_INT_W.IN_F) = 9.38 = 47 bits total.
    // mant_in has format 2.29 = MANT_MULT_I.MANT_MULT_F.
    // To place it such that the binary point lines up with the unified format:
    //   frac extra bits = IN_F - MANT_MULT_F = 38 - 29 = 9 bits padding below
    //   int bits free   = IN_CONV_INT_W - MANT_MULT_I = 9 - 2 = 7 bits above
    // So the initial aligned position is: {7'b0, mant_in, 9'b0} in 47-bit field.
    // -------------------------------------------------------------------------
    localparam int UNIFIED_W   = IN_CONV_INT_W + IN_F;   // 47
    localparam int FRAC_PAD    = IN_F - MANT_MULT_F;     // 38 - 29 = 9
    localparam int INT_PAD     = IN_CONV_INT_W - MANT_MULT_I;  // 9 - 2 = 7

    logic [UNIFIED_W-1:0] mant_aligned;

    // mant_in placed at bits [FRAC_PAD + MANT_MULT_W - 1 : FRAC_PAD]
    assign mant_aligned = UNIFIED_W'({mant_in, {FRAC_PAD{1'b0}}});

    // -------------------------------------------------------------------------
    // Barrel shifter: apply input exponent
    // Note: exponent range [-9..7] so shift range already bounded by INPUT_MIN/MAX_EXP
    // -------------------------------------------------------------------------
    logic [UNIFIED_W-1:0] unified_shifted;

    always_comb begin
        if ($signed(exponent) >= 0)
            unified_shifted = mant_aligned << exponent;
        else
            unified_shifted = mant_aligned >> (-exponent);
    end

    // -------------------------------------------------------------------------
    // Extract fractional and integer parts
    // -------------------------------------------------------------------------
    logic [IN_F-1:0]               frac_part_comb;
    logic signed [IN_CONV_INT_W-1:0] int_part_comb;

    assign frac_part_comb = unified_shifted[IN_F-1:0];
    // exponent_bias = -(int)val.to_int() = negate integer part
    assign int_part_comb  = -signed'(unified_shifted[UNIFIED_W-1 : IN_F]);

    generate
        if (REGISTER_OUTPUT) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    frac_part <= '0;
                    int_part  <= '0;
                end else begin
                    frac_part <= frac_part_comb;
                    int_part  <= int_part_comb;
                end
            end
        end else begin : gen_comb
            assign frac_part = frac_part_comb;
            assign int_part  = int_part_comb;
        end
    endgenerate

endmodule : bf16_unified_shift

// =============================================================================
// bf16_early_out.sv
// Detects special cases and early-out conditions for the exp2 pipeline.
//
// Logic (matches bf16_exp2.hpp):
//   NaN              -> qNaN indefinite
//   Zero             -> +1.0  (2^0 = 1)
//   +Inf             -> +1.0  (positive values always return 1)
//   -Inf             -> +0.0  (2^-inf = 0)
//   Positive x > 0   -> +1.0  (requirement: always 1 for x > 0)
//   Negative, exp<-9 -> +1.0  (near zero -> approx 1)
//   Negative, exp>7  -> +0.0  (large magnitude -> approx 0)
//   Otherwise        -> EO_NONE (use core approximation pipeline)
// =============================================================================

module bf16_early_out
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0
)(
    input  logic        clk,
    input  logic        rst_n,
    input  fp_raw_t     decomposed,
    output early_out_t  eo_code
);

    early_out_t eo_comb;

    always_comb begin
        eo_comb = EO_NONE;

        if (decomposed.status.is_nan) begin
            eo_comb = EO_QNAN;
        end else if (decomposed.status.is_zero) begin
            eo_comb = EO_PLUS_ONE;
        end else if (decomposed.status.is_inf) begin
            eo_comb = (decomposed.sign) ? EO_PLUS_ZERO : EO_PLUS_ONE;
        end else if (!decomposed.sign) begin
            // Positive normal or denormal
            eo_comb = EO_PLUS_ONE;
        end else begin
            // Negative normal or denormal
            if ($signed(decomposed.exponent) < $signed(9'(INPUT_MIN_EXP)))
                eo_comb = EO_PLUS_ONE;
            else if ($signed(decomposed.exponent) > $signed(9'(INPUT_MAX_EXP)))
                eo_comb = EO_PLUS_ZERO;
            else
                eo_comb = EO_NONE;
        end
    end

    generate
        if (REGISTER_OUTPUT) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) eo_code <= EO_PLUS_ONE;
                else        eo_code <= eo_comb;
            end
        end else begin : gen_comb
            assign eo_code = eo_comb;
        end
    endgenerate

endmodule : bf16_early_out

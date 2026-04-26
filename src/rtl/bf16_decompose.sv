// =============================================================================
// bf16_decompose.sv
// Decomposes a raw 16-bit BF16 value into sign, unbiased exponent, mantissa,
// hidden bit and status flags. Matches fp_decompose() in fp_utils.hpp.
// =============================================================================

module bf16_decompose
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] bf16_in,
    output fp_raw_t     decomposed
);

    logic       sign_bit;
    logic [7:0] raw_exp;
    logic [6:0] raw_mant;

    assign sign_bit = bf16_in[15];
    assign raw_exp  = bf16_in[14:7];
    assign raw_mant = bf16_in[6:0];

    logic exp_all_zero, exp_all_ones, mant_is_zero;
    assign exp_all_zero = (raw_exp  == 8'h00);
    assign exp_all_ones = (raw_exp  == 8'hFF);
    assign mant_is_zero = (raw_mant == 7'h00);

    fp_status_t status_comb;
    assign status_comb.is_zero     = exp_all_zero &  mant_is_zero;
    assign status_comb.is_denormal = exp_all_zero & ~mant_is_zero;
    assign status_comb.is_inf      = exp_all_ones &  mant_is_zero;
    assign status_comb.is_nan      = exp_all_ones & ~mant_is_zero;

    logic signed [8:0] exponent_comb;
    logic              hidden_bit_comb;

    always_comb begin
        if (status_comb.is_zero | status_comb.is_inf | status_comb.is_nan) begin
            exponent_comb   = 9'sd0;
            hidden_bit_comb = 1'b0;
        end else if (status_comb.is_denormal) begin
            exponent_comb   = 9'sd1 - 9'sd127;   // 1 - BIAS = -126
            hidden_bit_comb = 1'b0;
        end else begin
            exponent_comb   = signed'({1'b0, raw_exp}) - 9'sd127;
            hidden_bit_comb = 1'b1;
        end
    end

    fp_raw_t decomposed_comb;
    assign decomposed_comb.sign       = sign_bit;
    assign decomposed_comb.exponent   = exponent_comb;
    assign decomposed_comb.mantissa   = raw_mant;
    assign decomposed_comb.hidden_bit = hidden_bit_comb;
    assign decomposed_comb.status     = status_comb;

    generate
        if (REGISTER_OUTPUT) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) decomposed <= '0;
                else        decomposed <= decomposed_comb;
            end
        end else begin : gen_comb
            assign decomposed = decomposed_comb;
        end
    endgenerate

endmodule : bf16_decompose

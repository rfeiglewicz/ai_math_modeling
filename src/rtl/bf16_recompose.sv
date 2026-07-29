// =============================================================================
// bf16_recompose.sv
// Recomposes a fp_raw_t structure back into a raw 16-bit BF16 value.
// Matches fp_recompose() in fp_utils.hpp.
// =============================================================================

module bf16_recompose
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        pipe_en,
    input  fp_raw_t     components,
    output logic [15:0] bf16_out
);

    logic [15:0] bf16_comb;

    always_comb begin
        logic [7:0]        biased_exp;
        logic [6:0]        mant_out;
        logic signed [8:0] temp_exp;

        biased_exp = 8'h00;
        mant_out   = 7'h00;
        temp_exp   = 9'sd0;

        unique if (components.status.is_zero) begin
            biased_exp = 8'h00;
            mant_out   = 7'h00;
        end else if (components.status.is_inf) begin
            biased_exp = 8'hFF;
            mant_out   = 7'h00;
        end else if (components.status.is_nan) begin
            biased_exp = 8'hFF;
            mant_out   = (components.mantissa == 7'h00) ? 7'h40 : components.mantissa;
        end else if (components.status.is_denormal) begin
            biased_exp = 8'h00;
            mant_out   = components.mantissa;
        end else begin
            // Normal
            temp_exp = components.exponent + 9'sd127;
            if (temp_exp <= 9'sd0) begin
                biased_exp = 8'h00;
                mant_out   = 7'h00;
            end else if (temp_exp >= 9'sd255) begin
                biased_exp = 8'hFF;
                mant_out   = 7'h00;
            end else begin
                biased_exp = temp_exp[7:0];
                mant_out   = components.mantissa;
            end
        end

        bf16_comb = {components.sign, biased_exp, mant_out};
    end

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      bf16_out <= 16'h0000;
                else if (pipe_en) bf16_out <= bf16_comb;
            end
        end else if (REGISTER_OUTPUT) begin : gen_reg_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) bf16_out <= bf16_comb;
            end
        end else begin : gen_comb
            assign bf16_out = bf16_comb;
        end
    endgenerate

endmodule : bf16_recompose

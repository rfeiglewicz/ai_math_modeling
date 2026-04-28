// =============================================================================
// bf16_log2e_mult.sv
// Multiplies mantissa source (1.7 format) by log2(e) constant (1.22 format).
// Used only when base2=0 (exp(x) mode). When base2=1, output equals input.
//
// Parameterizable:
//   LOG2E_I               - integer bits of the constant (default: 1)
//   LOG2E_F               - fractional bits of the constant (default: 22)
//   LOG2E_VAL             - fixed-point value of log2(e) (default: 0x5c551d)
//
// NOTE: The output port mant_out provides the effective mantissa (either
// mant_src for base2 mode, or mant_mult for exp(x) mode). Both are provided
// so the top level does not require a MUX chain.
// =============================================================================

module bf16_log2e_mult
    import bf16_exp2_pkg::*;
#(
    parameter int LOG2E_I   = LOG2E_I_DEFAULT,    // 1
    parameter int LOG2E_F   = LOG2E_F_DEFAULT,    // 22
    parameter int LOG2E_VAL = LOG2E_VAL_DEFAULT,  // 0x5c551d
    parameter bit REGISTER_OUTPUT = 1'b0
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

    logic [MANT_MULT_W-1:0] mant_mult;
    logic [MANT_MULT_W-1:0] mant_out_comb;

    // Product: mant_src (1.7) * log2e (1.22) = 2.29 = 31 bits
    assign mant_mult = mant_src * LOG2E_VAL[LOG2E_W-1:0];

    // Mux: select base2 or base-e path
    // For base2 mode, zero-extend mant_src to MANT_MULT_W bits by left-shifting
    // to align in the 2.29 format (mant_src is 1.7, MANT_MULT_W is 31 = 2.29)
    // So mant_src (1.7) needs MANT_MULT_F - MANT_SRC_F = 29 - 7 = 22 extra bits
    localparam int ALIGN = MANT_MULT_F - MANT_SRC_F;  // = 22

    always_comb begin
        if (base2)
            mant_out_comb = MANT_MULT_W'({mant_src, {ALIGN{1'b0}}});
        else
            mant_out_comb = mant_mult;
    end

    generate
        if (REGISTER_OUTPUT) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      mant_out <= '0;
                else if (pipe_en) mant_out <= mant_out_comb;
            end
        end else begin : gen_comb
            assign mant_out = mant_out_comb;
        end
    endgenerate

endmodule : bf16_log2e_mult

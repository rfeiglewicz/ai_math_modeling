// =============================================================================
// bf16_normalize.sv
// Normalizes the polynomial result by finding the MSB (priority encoder),
// then applies a barrel-shift to align the MSB to the output format.
//
// C++ equivalent (bf16_exp2_poly):
//   msb_idx = find_msb(res_raw)             -- priority encoder
//   shift = (CALC_W - 1) - msb_idx          -- barrel shift amount
//   normalized = res_raw << shift
//   poly_mantissa = normalized[CALC_W-1 : CALC_W-POLY_OUT_W]
//   poly_exponent  = msb_idx - POLY_OUT_F
//
// Bit widths:
//   unnormalized_res : CALC_W = 62 bits (4.58 signed)
//   normalized_mant  : POLY_OUT_W = 59 bits (1.58 unsigned)
//   poly_exponent    : signed 9-bit (enough for -126..127)
// =============================================================================

module bf16_normalize
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1
)(
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      pipe_en,
    input  logic [CALC_W-1:0]        unnormalized_res,
    output logic [POLY_OUT_W-1:0]    normalized_mant,
    output logic signed [8:0]        poly_exponent
);

    logic signed [8:0]   msb_idx_comb;
    logic signed [8:0]   shift_amt;
    logic [CALC_W-1:0]   shifted_res;
    logic [POLY_OUT_W-1:0] poly_mant_comb;
    logic signed [8:0]   poly_exp_comb;

    // Priority encoder: find highest set bit
    always_comb begin
        msb_idx_comb = -9'sd1;
        for (int i = CALC_W - 1; i >= 0; --i) begin
            if (unnormalized_res[i]) begin
                msb_idx_comb = 9'(signed'(i));
                break;
            end
        end
    end

    // Barrel shift and slice
    always_comb begin
        shift_amt   = 9'(signed'(CALC_W - 1)) - msb_idx_comb;
        shifted_res = unnormalized_res << shift_amt;
        // Extract the POLY_OUT_W MSBs
        poly_mant_comb = shifted_res[CALC_W-1 : CALC_W-POLY_OUT_W];
        // Exponent relative to 1.POLY_OUT_F format
        poly_exp_comb  = msb_idx_comb - 9'(signed'(POLY_OUT_F));
    end

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    normalized_mant <= '0;
                    poly_exponent   <= '0;
                end else if (pipe_en) begin
                    normalized_mant <= poly_mant_comb;
                    poly_exponent   <= poly_exp_comb;
                end
            end
        end else if (REGISTER_OUTPUT) begin : gen_reg_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    normalized_mant <= poly_mant_comb;
                    poly_exponent   <= poly_exp_comb;
                end
            end
        end else begin : gen_comb
            assign normalized_mant = poly_mant_comb;
            assign poly_exponent   = poly_exp_comb;
        end
    endgenerate

endmodule : bf16_normalize

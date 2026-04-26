// =============================================================================
// bf16_exp2.sv
// Top-level module for the BF16 exp2 / expe approximation pipeline.
//
// Architecture (8 functional sub-modules):
//   1. bf16_decompose     - BF16 bit field decomposition
//   2. bf16_early_out     - Special case and range check detector
//   3. bf16_log2e_mult    - Optional log2(e) multiply for exp(x) mode
//   4. bf16_unified_shift - Input exponent scaling, int/frac split
//   5. bf16_linear_approx - ROM-based piecewise linear approximation
//   6. bf16_normalize     - Priority encoder + barrel shifter
//   7. bf16_round         - RNE rounding and output assembly
//   8. bf16_recompose     - Output FP recomposition to 16-bit BF16
//
// Pipeline notes:
//   When REGISTER_STAGES=1, each sub-module inserts one register at its output.
//   The eo_code signal is propagated through delay registers separately to
//   stay time-aligned with the core path results before the final mux.
//
// Parameterizable:
//   REGISTER_STAGES  - 1 = add output registers to all sub-modules
//   LOG2E_I/F/VAL    - log2(e) constant format (passed to bf16_log2e_mult)
//   COEFF_W/F        - coefficient format (passed to bf16_linear_approx)
// =============================================================================

module bf16_exp2
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b0,   // Enable per-stage registers
    parameter int LOG2E_I    = LOG2E_I_DEFAULT,
    parameter int LOG2E_F    = LOG2E_F_DEFAULT,
    parameter int LOG2E_VAL  = LOG2E_VAL_DEFAULT,
    parameter int COEFF_W    = COEFF_W_DEFAULT,
    parameter int COEFF_F    = COEFF_F_DEFAULT
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] bf16_in,
    input  logic        base2,   // 1 = 2^x mode;  0 = e^x mode
    output logic [15:0] bf16_out
);

    // =========================================================================
    // Stage 1: Decompose
    // =========================================================================
    fp_raw_t s1_decomposed;

    bf16_decompose #(
        .REGISTER_OUTPUT(REGISTER_STAGES)
    ) u_decompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .bf16_in   (bf16_in),
        .decomposed(s1_decomposed)
    );

    // =========================================================================
    // Stage 2: Early-out detection
    // =========================================================================
    early_out_t s2_eo_code;

    bf16_early_out #(
        .REGISTER_OUTPUT(REGISTER_STAGES)
    ) u_early_out (
        .clk       (clk),
        .rst_n     (rst_n),
        .decomposed(s1_decomposed),
        .eo_code   (s2_eo_code)
    );

    // =========================================================================
    // Pipeline alignment registers (active only when REGISTER_STAGES=1).
    //
    // Problem: some signals skip stages and arrive 1+ cycles too early:
    //   - base2 (raw port) is used in stage 3 alongside s1_decomposed (T+1)
    //     → needs 1 extra FF so both arrive at T+1
    //   - s1_decomposed.exponent is used in stage 4 alongside s3_mant_out (T+2)
    //     → needs 1 extra FF after stage-1 output so it arrives at T+2
    //   - s4_int_part is used in stage 7 alongside s6_norm_mant (T+5)
    //     → needs 2 extra FFs (T+3 → T+4 → T+5)
    // =========================================================================
    logic                            s3_base2;      // base2 delayed to match s1_decomposed
    logic signed [8:0]               s4_exponent;  // exponent aligned with s3_mant_out
    logic signed [IN_CONV_INT_W-1:0] s5_int_part;  // int_part delayed +1
    logic signed [IN_CONV_INT_W-1:0] s6_int_part;  // int_part delayed +2 (aligned with s6_poly_exp)

    generate
        if (REGISTER_STAGES) begin : gen_align_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s3_base2    <= 1'b1;
                    s4_exponent <= '0;
                end else begin
                    s3_base2    <= base2;
                    s4_exponent <= s1_decomposed.exponent;
                end
            end
        end else begin : gen_align_wire
            assign s3_base2    = base2;
            assign s4_exponent = s1_decomposed.exponent;
        end
    endgenerate

    // =========================================================================
    // Stage 3: Log2(e) multiply (for expe mode; bypass in base2 mode)
    // Input: s1_decomposed.mantissa + hidden bit (both at T+1)
    // =========================================================================
    logic [MANT_SRC_W-1:0]  s3_mant_src;    // {hidden_bit, mantissa}
    logic [MANT_MULT_W-1:0] s3_mant_out;

    assign s3_mant_src = {s1_decomposed.hidden_bit, s1_decomposed.mantissa};

    bf16_log2e_mult #(
        .LOG2E_I         (LOG2E_I),
        .LOG2E_F         (LOG2E_F),
        .LOG2E_VAL       (LOG2E_VAL),
        .REGISTER_OUTPUT (REGISTER_STAGES)
    ) u_log2e_mult (
        .clk      (clk),
        .rst_n    (rst_n),
        .base2    (s3_base2),      // delayed to match mant_src latency
        .mant_src (s3_mant_src),
        .mant_out (s3_mant_out)
    );

    // =========================================================================
    // Stage 4: Unified shift (input exponent alignment)
    // =========================================================================
    logic [IN_F-1:0]                     s4_frac_part;
    logic signed [IN_CONV_INT_W-1:0]     s4_int_part;

    bf16_unified_shift #(
        .REGISTER_OUTPUT(REGISTER_STAGES)
    ) u_unified_shift (
        .clk      (clk),
        .rst_n    (rst_n),
        .mant_in  (s3_mant_out),
        .exponent (s4_exponent),   // delayed to match s3_mant_out latency
        .frac_part(s4_frac_part),
        .int_part (s4_int_part)
    );

    // =========================================================================
    // Stage 5: Piecewise linear approximation
    // =========================================================================
    logic [CALC_W-1:0] s5_unnorm_res;

    bf16_linear_approx #(
        .COEFF_W         (COEFF_W),
        .COEFF_F         (COEFF_F),
        .REGISTER_OUTPUT (REGISTER_STAGES)
    ) u_lin_approx (
        .clk             (clk),
        .rst_n           (rst_n),
        .frac_part       (s4_frac_part),
        .unnormalized_res(s5_unnorm_res)
    );

    // =========================================================================
    // Stage 6: Normalize
    // =========================================================================
    logic [POLY_OUT_W-1:0] s6_norm_mant;
    logic signed [8:0]     s6_poly_exp;

    bf16_normalize #(
        .REGISTER_OUTPUT(REGISTER_STAGES)
    ) u_normalize (
        .clk             (clk),
        .rst_n           (rst_n),
        .unnormalized_res(s5_unnorm_res),
        .normalized_mant (s6_norm_mant),
        .poly_exponent   (s6_poly_exp)
    );

    // =========================================================================
    // int_part alignment: s4_int_part (T+3) must reach stage 7 at T+5.
    // Pipeline adds registers in stages 5 and 6, so we need 2 extra FFs.
    // =========================================================================
    generate
        if (REGISTER_STAGES) begin : gen_int_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s5_int_part <= '0;
                    s6_int_part <= '0;
                end else begin
                    s5_int_part <= s4_int_part;
                    s6_int_part <= s5_int_part;
                end
            end
        end else begin : gen_int_wire
            assign s5_int_part = s4_int_part;
            assign s6_int_part = s4_int_part;
        end
    endgenerate

    // =========================================================================
    // Stage 7: Round (RNE)
    // =========================================================================
    fp_raw_t s7_rounded_fp;

    bf16_round #(
        .REGISTER_OUTPUT(REGISTER_STAGES)
    ) u_round (
        .clk          (clk),
        .rst_n        (rst_n),
        .poly_mantissa(s6_norm_mant),
        .poly_exponent(s6_poly_exp),
        .exponent_bias(s6_int_part),   // delayed 2 extra cycles to align with s6_poly_exp
        .rounded_fp   (s7_rounded_fp)
    );

    // =========================================================================
    // Delay chain for eo_code to keep it aligned with s7_rounded_fp.
    // Pipeline latency breakdown (REGISTER_STAGES=1):
    //   s2_eo_code is valid at T+2 (after stage1 + stage2 registers).
    //   s7_rounded_fp is valid at T+6 (stages 1,3,4,5,6,7 registers).
    //   → EO_DELAY_STAGES = 6 - 2 = 4 extra cycles.
    // With REGISTER_STAGES=0: no delay needed (all combinational).
    // =========================================================================
    localparam int EO_DELAY_STAGES = REGISTER_STAGES ? 4 : 0;

    early_out_t eo_delay [EO_DELAY_STAGES+1];
    assign eo_delay[0] = s2_eo_code;

    generate
        for (genvar i = 0; i < EO_DELAY_STAGES; i++) begin : gen_eo_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) eo_delay[i+1] <= EO_PLUS_ONE;
                else        eo_delay[i+1] <= eo_delay[i];
            end
        end
    endgenerate

    early_out_t eo_aligned;
    assign eo_aligned = eo_delay[EO_DELAY_STAGES];

    // =========================================================================
    // Stage 8: Apply early-out mux and recompose
    // =========================================================================
    fp_raw_t s8_final_fp;

    always_comb begin
        unique case (eo_aligned)
            EO_QNAN: begin
                s8_final_fp              = '0;
                s8_final_fp.status.is_nan = 1'b1;
                s8_final_fp.sign          = 1'b1;
                s8_final_fp.mantissa      = 7'b100_0000;  // qNaN indefinite: MSB of mant set
            end
            EO_PLUS_ONE: begin
                s8_final_fp              = '0;
                s8_final_fp.exponent     = 9'sd0;     // unbiased 0 = biased 127 -> 3f80
                s8_final_fp.hidden_bit   = 1'b1;
            end
            EO_PLUS_ZERO: begin
                s8_final_fp              = '0;
                s8_final_fp.status.is_zero = 1'b1;
            end
            default: begin  // EO_NONE
                s8_final_fp = s7_rounded_fp;
            end
        endcase
    end

    bf16_recompose #(
        .REGISTER_OUTPUT(REGISTER_STAGES)
    ) u_recompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .components(s8_final_fp),
        .bf16_out  (bf16_out)
    );

endmodule : bf16_exp2

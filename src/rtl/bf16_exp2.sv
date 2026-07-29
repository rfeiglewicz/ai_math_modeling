// =============================================================================
// bf16_exp2.sv
// Top-level module for the BF16 exp2 / expe approximation pipeline.
// AXI-Stream interface: s_axis (slave/input) and m_axis (master/output).
//
// Architecture (8 functional sub-modules):
//   1. bf16_decompose     - BF16 bit field decomposition
//   2. bf16_early_out     - Special case and range check detector
//   3. bf16_log2e_mult    - Optional log2(e) multiply for exp(x) mode
//   4. bf16_unified_shift - Input exponent scaling, int/frac split
//   5. bf16_linear_approx - ROM-based piecewise linear approximation (BRAM)
//   6. bf16_normalize     - Priority encoder + barrel shifter
//   7. bf16_round         - RNE rounding and output assembly
//   8. bf16_recompose     - Output FP recomposition to 16-bit BF16
//
// Pipeline notes:
//   When REGISTER_STAGES=1, each sub-module inserts one register at its output.
//   pipe_en signal gates ALL pipeline registers for AXI-Stream backpressure.
//   When m_axis_tready=0 and m_axis_tvalid=1, the entire pipeline freezes.
//
// Pipeline depth (REGISTER_STAGES=1): 7 FF stages -> 6 cycle latency.
// Pipeline depth (REGISTER_STAGES=0): 0 (fully combinational, single-cycle).
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
    parameter int MANT_MULT_ROUND_FRAC = MANT_MULT_F,  // RNE rounding after log2e mult (29=full)
    parameter int COEFF_W    = COEFF_W_DEFAULT,
    parameter int COEFF_F    = COEFF_F_DEFAULT,
    // 0 = fabric barrel shifter, 1 = one-hot multiply on DSP48 (bit-exact)
    parameter bit DSP_SHIFT  = 1'b0,
    // 1 = asynchronous reset on datapath registers.
    // 0 = datapath registers have no reset. This is safe because the output is
    //     only presented when m_axis_tvalid is high, and that comes from vld_sr
    //     which IS reset -- stale datapath values are always flagged invalid.
    //     Dropping the reset lets Vivado absorb registers into DSP48/BRAM.
    parameter bit RESET_DATAPATH = 1'b1,
    // Pad the output to this many pipeline stages so every core has the same
    // latency. 0 = natural depth. See bf16_exp2_pkg::UNIFIED_PIPE_DEPTH.
    parameter int PIPE_TARGET = UNIFIED_PIPE_DEPTH
)(
    input  logic        clk,
    input  logic        rst_n,

    // Slave AXI-Stream (input)
    input  logic [15:0] s_axis_tdata,    // BF16 input value
    input  logic        s_axis_tuser,    // 1 = 2^x (base2), 0 = e^x
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,

    // Master AXI-Stream (output)
    output logic [15:0] m_axis_tdata,    // BF16 result
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready
);

    // =========================================================================
    // Pipeline depth: 7 register stages when REGISTER_STAGES=1.
    // Data path FFs: decompose, log2e_mult, unified_shift, BRAM(linear_approx),
    //                normalize, round, recompose.
    // =========================================================================
    localparam int CORE_DEPTH = REGISTER_STAGES ? 7 : 0;
    localparam int PAD_STAGES = (REGISTER_STAGES && PIPE_TARGET > CORE_DEPTH)
                                ? PIPE_TARGET - CORE_DEPTH : 0;
    localparam int PIPE_DEPTH = CORE_DEPTH + PAD_STAGES;

    // =========================================================================
    // AXI-Stream handshake & pipeline enable
    //
    // "All-stall" approach: when downstream cannot accept (m_axis_tready=0)
    // and the output is valid, the ENTIRE pipeline freezes.
    //
    // pipe_en=1 => pipeline advances (all FFs capture new values)
    // pipe_en=0 => pipeline holds    (all FFs retain their values)
    // =========================================================================
    logic pipe_en;

    generate
        if (REGISTER_STAGES) begin : gen_axi_pipelined
            // --- Valid shift register (PIPE_DEPTH stages) ---
            logic [PIPE_DEPTH-1:0] vld_sr;

            assign m_axis_tvalid = vld_sr[PIPE_DEPTH-1];
            assign pipe_en       = m_axis_tready | ~m_axis_tvalid;
            assign s_axis_tready = pipe_en;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    vld_sr <= '0;
                else if (pipe_en)
                    vld_sr <= {vld_sr[PIPE_DEPTH-2:0], s_axis_tvalid};
            end
        end else begin : gen_axi_comb
            // Combinational mode: data passes through in one cycle
            assign m_axis_tvalid = s_axis_tvalid;
            assign s_axis_tready = m_axis_tready;
            assign pipe_en       = 1'b1;
        end
    endgenerate

    // =========================================================================
    // Internal wiring from AXI ports to datapath
    // =========================================================================
    logic [15:0] bf16_in;
    logic        base2;

    assign bf16_in = s_axis_tdata;
    assign base2   = s_axis_tuser;

    // =========================================================================
    // Stage 1: Decompose
    // =========================================================================
    fp_raw_t s1_decomposed;

    bf16_decompose #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH)
    ) u_decompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .pipe_en   (pipe_en),
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
        .pipe_en   (pipe_en),
        .decomposed(s1_decomposed),
        .eo_code   (s2_eo_code)
    );

    // =========================================================================
    // Pipeline alignment registers (active only when REGISTER_STAGES=1).
    //
    //   - base2 (raw port) needs 1 extra FF to match s1_decomposed (T+1)
    //   - exponent needs 1 extra FF to match s3_mant_out (T+2)
    //   - int_part needs 2 extra FFs to match s6 outputs (T+5)
    //
    // All alignment FFs are gated by pipe_en.
    // =========================================================================
    logic                            s3_base2;
    logic signed [8:0]               s4_exponent;
    logic signed [IN_CONV_INT_W-1:0] s5_int_part;
    logic signed [IN_CONV_INT_W-1:0] s6_int_part;

    generate
        if (REGISTER_STAGES && RESET_DATAPATH) begin : gen_align_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s3_base2    <= 1'b1;
                    s4_exponent <= '0;
                end else if (pipe_en) begin
                    s3_base2    <= base2;
                    s4_exponent <= s1_decomposed.exponent;
                end
            end
        end else if (REGISTER_STAGES) begin : gen_align_regs_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) begin
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
    // =========================================================================
    logic [MANT_SRC_W-1:0]  s3_mant_src;
    logic [MANT_MULT_W-1:0] s3_mant_out;

    assign s3_mant_src = {s1_decomposed.hidden_bit, s1_decomposed.mantissa};

    bf16_log2e_mult #(
        .LOG2E_I              (LOG2E_I),
        .LOG2E_F              (LOG2E_F),
        .LOG2E_VAL            (LOG2E_VAL),
        .MANT_MULT_ROUND_FRAC(MANT_MULT_ROUND_FRAC),
        .REGISTER_OUTPUT      (REGISTER_STAGES),
        .RESET_DATAPATH       (RESET_DATAPATH)
    ) u_log2e_mult (
        .clk      (clk),
        .rst_n    (rst_n),
        .pipe_en  (pipe_en),
        .base2    (s3_base2),
        .mant_src (s3_mant_src),
        .mant_out (s3_mant_out)
    );

    // =========================================================================
    // Stage 4: Unified shift (input exponent alignment)
    // =========================================================================
    logic [IN_F-1:0]                     s4_frac_part;
    logic signed [IN_CONV_INT_W-1:0]     s4_int_part;

    bf16_unified_shift #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .DSP_SHIFT      (DSP_SHIFT),
        .RESET_DATAPATH (RESET_DATAPATH)
    ) u_unified_shift (
        .clk      (clk),
        .rst_n    (rst_n),
        .pipe_en  (pipe_en),
        .mant_in  (s3_mant_out),
        .exponent (s4_exponent),
        .frac_part(s4_frac_part),
        .int_part (s4_int_part)
    );

    // =========================================================================
    // Stage 5: Piecewise linear approximation (BRAM coefficient ROM)
    // =========================================================================
    logic [CALC_W-1:0] s5_unnorm_res;

    bf16_linear_approx #(
        .COEFF_W         (COEFF_W),
        .COEFF_F         (COEFF_F),
        .REGISTER_OUTPUT (REGISTER_STAGES),
        .RESET_DATAPATH  (RESET_DATAPATH),
        .FRAC_ZERO_LSBS  (MANT_MULT_F - MANT_MULT_ROUND_FRAC)
    ) u_lin_approx (
        .clk             (clk),
        .rst_n           (rst_n),
        .pipe_en         (pipe_en),
        .frac_part       (s4_frac_part),
        .unnormalized_res(s5_unnorm_res)
    );

    // =========================================================================
    // Stage 6: Normalize
    // =========================================================================
    logic [POLY_OUT_W-1:0] s6_norm_mant;
    logic signed [8:0]     s6_poly_exp;

    bf16_normalize #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH)
    ) u_normalize (
        .clk             (clk),
        .rst_n           (rst_n),
        .pipe_en         (pipe_en),
        .unnormalized_res(s5_unnorm_res),
        .normalized_mant (s6_norm_mant),
        .poly_exponent   (s6_poly_exp)
    );

    // =========================================================================
    // int_part alignment: s4_int_part (T+3) must reach stage 7 at T+5.
    // Pipeline adds registers in stages 5 and 6, so we need 2 extra FFs.
    // =========================================================================
    generate
        if (REGISTER_STAGES && RESET_DATAPATH) begin : gen_int_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s5_int_part <= '0;
                    s6_int_part <= '0;
                end else if (pipe_en) begin
                    s5_int_part <= s4_int_part;
                    s6_int_part <= s5_int_part;
                end
            end
        end else if (REGISTER_STAGES) begin : gen_int_delay_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) begin
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
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH)
    ) u_round (
        .clk          (clk),
        .rst_n        (rst_n),
        .pipe_en      (pipe_en),
        .poly_mantissa(s6_norm_mant),
        .poly_exponent(s6_poly_exp),
        .exponent_bias(s6_int_part),
        .rounded_fp   (s7_rounded_fp)
    );

    // =========================================================================
    // Delay chain for eo_code to keep it aligned with s7_rounded_fp.
    // EO_DELAY_STAGES = 4 when REGISTER_STAGES=1 (eo at T+2, data at T+6).
    // =========================================================================
    localparam int EO_DELAY_STAGES = REGISTER_STAGES ? 4 : 0;

    early_out_t eo_delay [EO_DELAY_STAGES+1];
    assign eo_delay[0] = s2_eo_code;

    generate
        for (genvar i = 0; i < EO_DELAY_STAGES; i++) begin : gen_eo_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      eo_delay[i+1] <= EO_PLUS_ONE;
                else if (pipe_en) eo_delay[i+1] <= eo_delay[i];
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
                s8_final_fp.mantissa      = 7'b100_0000;
            end
            EO_PLUS_ONE: begin
                s8_final_fp              = '0;
                s8_final_fp.exponent     = 9'sd0;
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

    logic [15:0] bf16_out;

    bf16_recompose #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH)
    ) u_recompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .pipe_en   (pipe_en),
        .components(s8_final_fp),
        .bf16_out  (bf16_out)
    );

    // Latency padding so this core matches the deepest implementation.
    bf16_pipe_pad #(
        .STAGES        (PAD_STAGES),
        .WIDTH         (16),
        .RESET_DATAPATH(RESET_DATAPATH)
    ) u_pipe_pad (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .d(bf16_out), .q(m_axis_tdata)
    );

endmodule : bf16_exp2

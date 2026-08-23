// =============================================================================
// bf16_exp2_optim.sv
// Top level for the width-optimised BF16 exp2 / expe pipeline.
// AXI-Stream interface, pin-compatible and latency-compatible with bf16_exp2.
//
// Same algorithm, same results, narrower datapath. Every width comes from
// bf16_exp2_optim_pkg and is the smallest that keeps the core bit-identical
// to bf16_exp2_approx<29> over all 65536 BF16 patterns in both modes.
//
// Architecture (8 functional sub-modules):
//   1. bf16_decompose               - BF16 bit field decomposition   (shared)
//   2. bf16_early_out               - special case / range detector  (shared)
//   3. bf16_exp2_optim_log2e_mult   - log2(e) multiply, rounded to Q2.21
//   4. bf16_exp2_optim_shift        - exponent shift, int/frac split
//   5. bf16_exp2_optim_approx       - 128 x 35 b ROM, 17x17 multiply, 18 b subtract
//   6. bf16_exp2_optim_normalize    - one bit test and a 2:1 mux
//   7. bf16_exp2_optim_round        - RNE rounding and output assembly
//   8. bf16_recompose               - FP recomposition to 16-bit BF16  (shared)
//
// Four modules are shared with the production core unmodified: decompose,
// early_out, recompose and pipe_pad. They are BF16 format plumbing, identical
// in both designs, and carry no arithmetic width. Everything on the datapath
// is new.
//
// -----------------------------------------------------------------------------
// What changed against bf16_exp2
//
//   stage               production   here    saved
//   log2(e) product     31 b         23 b       8
//   unified shift reg   47 b         30 b      17
//   polynomial input x  38 b         17 b      21
//   coefficient a       21 b         17 b       4
//   coefficient b       21 b         18 b       3
//   a * x product       60 b         34 b      26
//   b - a*x adder       62 b         18 b      44
//   normalised mantissa 59 b         19 b      40
//   datapath total     339 b        176 b     163
//   coefficient ROM   5376 b       4480 b     896
//
//   multiplier shape   21 x 38  ->  17 x 17   (two cascaded DSP48 -> one)
//
// Three pieces of hardware are gone rather than merely narrower: the sign bit
// and integer guard bits of the accumulator, the 62-bit priority encoder and
// the 62-bit barrel shifter. All three follow from b - a*x landing in
// [0.5, 1) for every reachable input. See bf16_exp2_optim_normalize.sv for
// the caveat and why the guarded variant is the default.
//
// -----------------------------------------------------------------------------
// Pipeline
//
// REGISTER_STAGES=1 gives 7 base FF stages plus the retiming registers; the
// output is then padded to PIPE_TARGET so the latency matches every other
// core in this repository. The narrow datapath needs less retiming than the
// production core does (3 extra stages instead of 9), so more of the budget
// is padding -- which is free to remove if this core is used on its own.
//
// REGISTER_STAGES=0 is fully combinational, single cycle.
// =============================================================================

module bf16_exp2_optim
    import bf16_exp2_optim_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b0,
    parameter int LOG2E_I    = LOG2E_I_DEFAULT,
    parameter int LOG2E_F    = LOG2E_F_DEFAULT,
    parameter int LOG2E_VAL  = LOG2E_VAL_DEFAULT,
    // Fractional bits kept after rounding the log2(e) product. 21 is the
    // verified minimum for RNE; lower values are expected to fail and the
    // rtl_optim_round_frac_sweep target checks that from the RTL side.
    parameter int MANT_MULT_ROUND_FRAC = MANT_MULT_F,
    // 0 = RNE, 1 = round-half-up, 2 = truncate.
    //
    // RNE is the default because it is what the C++ model does, so the two
    // stay literally the same arithmetic. But half-up reaches the same 21-bit
    // minimum and synthesises better: with the retiming maxed out it moves the
    // critical path off this multiply and buys 164.8 -> 177.5 MHz for one LUT.
    // If Fmax matters more than model symmetry, build with ROUND_MODE=1; it is
    // covered by make rtl_optim_round_frac_sweep OPTIM_ROUND_MODE=1.
    // Truncation needs 22 bits and cannot be used here at all.
    parameter int ROUND_MODE  = 0,
    // 1 = assume b - a*x >= 0.5 and shift by a constant.
    // 0 = test the top bit and shift by 1 or 2. Same width, same results.
    parameter bit STATIC_NORM = 1'b0,
    // 1 = asynchronous reset on datapath registers.
    // 0 = datapath registers have no reset. Safe because the output is only
    //     presented when m_axis_tvalid is high, and that comes from vld_sr
    //     which IS reset. Dropping it lets Vivado absorb registers into
    //     DSP48/BRAM.
    parameter bit RESET_DATAPATH = 1'b1,
    // Pad the output to this many pipeline stages so every core in the repo
    // has the same latency. 0 = natural depth.
    parameter int PIPE_TARGET  = bf16_exp2_pkg::UNIFIED_PIPE_DEPTH,
    // -GREPORT_DEPTH=1 prints the resulting pipeline depth at elaboration.
    parameter bit REPORT_DEPTH = 1'b0,
    // -------------------------------------------------------------------------
    // Retiming knobs, same meaning as in bf16_exp2 but with smaller defaults:
    // this datapath has an 8x23 multiply, a 30-bit shifter, a 17x17 multiply
    // and an 18-bit subtract, none of which need to be split as hard as the
    // 21x30-into-62-bit chain in the production core.
    //
    //   RETIME_LOG2E  0..2  registers inside the log2(e) multiply
    //   RETIME_SHIFT  0..1  register inside the exponent shift
    //   RETIME_APPROX 0..2  registers inside the linear approximation
    //   RETIME_NORM   0..1  register inside the normaliser (a mux; rarely useful)
    //   RETIME_ROUND  0..2  registers inside the rounder
    // -------------------------------------------------------------------------
    parameter int RETIME_LOG2E  = 1,
    parameter int RETIME_SHIFT  = 0,
    parameter int RETIME_APPROX = 1,
    parameter int RETIME_NORM   = 0,
    parameter int RETIME_ROUND  = 1
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
    // Pipeline depth bookkeeping. Identical structure to bf16_exp2 so the two
    // cores can be compared stage for stage.
    // =========================================================================
    localparam int RT_LOG2E  = REGISTER_STAGES ? RETIME_LOG2E  : 0;
    localparam int RT_SHIFT  = REGISTER_STAGES ? RETIME_SHIFT  : 0;
    localparam int RT_APPROX = REGISTER_STAGES ? RETIME_APPROX : 0;
    localparam int RT_NORM   = REGISTER_STAGES ? RETIME_NORM   : 0;
    localparam int RT_ROUND  = REGISTER_STAGES ? RETIME_ROUND  : 0;

    // Registers added before the rounder shift both the int_part and the
    // early-out code; registers added inside the rounder shift only the
    // early-out code, because int_part is consumed at the rounder input.
    // A register inside the log2(e) multiply is downstream of the exponent
    // and upstream of int_part, so it lengthens the exponent and early-out
    // chains only; the same goes for the shift stage.
    localparam int RETIME_PRE   = RT_APPROX + RT_NORM;
    localparam int RETIME_POST  = RETIME_PRE + RT_ROUND;
    localparam int RETIME_EXTRA = RT_LOG2E + RT_SHIFT + RETIME_POST;
    localparam int CORE_DEPTH   = REGISTER_STAGES ? (7 + RETIME_EXTRA) : 0;
    localparam int PAD_STAGES   = (REGISTER_STAGES && PIPE_TARGET > CORE_DEPTH)
                                  ? PIPE_TARGET - CORE_DEPTH : 0;
    localparam int PIPE_DEPTH   = CORE_DEPTH + PAD_STAGES;

    if (REPORT_DEPTH) begin : gen_depth_report
        $info("CORE_PIPE_DEPTH core=%0d pad=%0d total=%0d",
              CORE_DEPTH, PAD_STAGES, PIPE_DEPTH);
    end

    // =========================================================================
    // AXI-Stream handshake & pipeline enable
    //
    // "All-stall": when downstream cannot accept and the output is valid, the
    // ENTIRE pipeline freezes.
    // =========================================================================
    logic pipe_en;

    generate
        if (REGISTER_STAGES) begin : gen_axi_pipelined
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
            assign m_axis_tvalid = s_axis_tvalid;
            assign s_axis_tready = m_axis_tready;
            assign pipe_en       = 1'b1;
        end
    endgenerate

    logic [15:0] bf16_in;
    logic        base2;

    assign bf16_in = s_axis_tdata;
    assign base2   = s_axis_tuser;

    // =========================================================================
    // Stage 1: Decompose  (shared with bf16_exp2, unmodified)
    // =========================================================================
    bf16_exp2_pkg::fp_raw_t s1_decomposed;

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
    // Stage 2: Early-out detection  (shared with bf16_exp2, unmodified)
    // =========================================================================
    bf16_exp2_pkg::early_out_t s2_eo_code;

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
    // Alignment registers
    //   - base2    needs 1 extra FF to match s1_decomposed
    //   - exponent needs 1 + RT_LOG2E FFs to match the log2(e) output
    // =========================================================================
    logic                            s3_base2;
    logic signed [8:0]               s4_exponent;
    logic signed [IN_CONV_INT_W-1:0] s6_int_part;

    localparam int EXP_DELAY = REGISTER_STAGES ? (1 + RT_LOG2E) : 0;

    logic signed [8:0] exp_delay [EXP_DELAY+1];
    assign exp_delay[0] = s1_decomposed.exponent;

    generate
        if (REGISTER_STAGES && RESET_DATAPATH) begin : gen_align_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       s3_base2 <= 1'b1;
                else if (pipe_en) s3_base2 <= base2;
            end
        end else if (REGISTER_STAGES) begin : gen_align_regs_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) s3_base2 <= base2;
            end
        end else begin : gen_align_wire
            assign s3_base2 = base2;
        end

        for (genvar i = 0; i < EXP_DELAY; i++) begin : gen_exp_delay
            if (RESET_DATAPATH) begin : gen_rst
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)       exp_delay[i+1] <= '0;
                    else if (pipe_en) exp_delay[i+1] <= exp_delay[i];
                end
            end else begin : gen_nrst
                always_ff @(posedge clk) begin
                    if (pipe_en) exp_delay[i+1] <= exp_delay[i];
                end
            end
        end
    endgenerate

    assign s4_exponent = exp_delay[EXP_DELAY];

    // =========================================================================
    // Stage 3: log2(e) multiply, rounded straight to Q2.21
    // =========================================================================
    logic [MANT_SRC_W-1:0]  s3_mant_src;
    logic [MANT_MULT_W-1:0] s3_mant_out;

    assign s3_mant_src = {s1_decomposed.hidden_bit, s1_decomposed.mantissa};

    bf16_exp2_optim_log2e_mult #(
        .LOG2E_I             (LOG2E_I),
        .LOG2E_F             (LOG2E_F),
        .LOG2E_VAL           (LOG2E_VAL),
        .MANT_MULT_ROUND_FRAC(MANT_MULT_ROUND_FRAC),
        .ROUND_MODE          (ROUND_MODE),
        .REGISTER_OUTPUT     (REGISTER_STAGES),
        .RESET_DATAPATH      (RESET_DATAPATH),
        .EXTRA_STAGES        (RT_LOG2E)
    ) u_log2e_mult (
        .clk     (clk),
        .rst_n   (rst_n),
        .pipe_en (pipe_en),
        .base2   (s3_base2),
        .mant_src(s3_mant_src),
        .mant_out(s3_mant_out)
    );

    // =========================================================================
    // Stage 4: exponent shift and int/frac split
    // =========================================================================
    logic [X_W-1:0]                  s4_x;
    logic signed [IN_CONV_INT_W-1:0] s4_int_part;

    bf16_exp2_optim_shift #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH),
        .EXTRA_STAGES   (RT_SHIFT)
    ) u_shift (
        .clk     (clk),
        .rst_n   (rst_n),
        .pipe_en (pipe_en),
        .mant_in (s3_mant_out),
        .exponent(s4_exponent),
        .x       (s4_x),
        .int_part(s4_int_part)
    );

    // =========================================================================
    // Stage 5: piecewise linear approximation
    // =========================================================================
    logic [CALC_W-1:0] s5_res;

    bf16_exp2_optim_approx #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH),
        .EXTRA_STAGES   (RT_APPROX)
    ) u_lin_approx (
        .clk    (clk),
        .rst_n  (rst_n),
        .pipe_en(pipe_en),
        .x      (s4_x),
        .res    (s5_res)
    );

    // =========================================================================
    // Stage 6: normalise
    // =========================================================================
    logic [POLY_OUT_W-1:0] s6_norm_mant;
    logic signed [8:0]     s6_poly_exp;

    bf16_exp2_optim_normalize #(
        .STATIC_NORM    (STATIC_NORM),
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH),
        .EXTRA_STAGES   (RT_NORM)
    ) u_normalize (
        .clk            (clk),
        .rst_n          (rst_n),
        .pipe_en        (pipe_en),
        .res            (s5_res),
        .normalized_mant(s6_norm_mant),
        .poly_exponent  (s6_poly_exp)
    );

    // =========================================================================
    // int_part alignment: it leaves the shift stage and has to meet the
    // rounder again after the approximation and normalisation stages.
    // =========================================================================
    localparam int INT_DELAY = REGISTER_STAGES ? (2 + RETIME_PRE) : 0;

    logic signed [IN_CONV_INT_W-1:0] int_delay [INT_DELAY+1];
    assign int_delay[0] = s4_int_part;

    generate
        for (genvar i = 0; i < INT_DELAY; i++) begin : gen_int_delay
            if (RESET_DATAPATH) begin : gen_rst
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)       int_delay[i+1] <= '0;
                    else if (pipe_en) int_delay[i+1] <= int_delay[i];
                end
            end else begin : gen_nrst
                always_ff @(posedge clk) begin
                    if (pipe_en) int_delay[i+1] <= int_delay[i];
                end
            end
        end
    endgenerate

    assign s6_int_part = int_delay[INT_DELAY];

    // =========================================================================
    // Stage 7: round (RNE)
    // =========================================================================
    bf16_exp2_pkg::fp_raw_t s7_rounded_fp;

    bf16_exp2_optim_round #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .RESET_DATAPATH (RESET_DATAPATH),
        .EXTRA_STAGES   (RT_ROUND)
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
    // Early-out alignment: eo is ready at T+2, the datapath at T+6 plus the
    // retiming registers.
    // =========================================================================
    localparam int EO_DELAY_STAGES = REGISTER_STAGES ? (4 + RETIME_EXTRA) : 0;

    bf16_exp2_pkg::early_out_t eo_delay [EO_DELAY_STAGES+1];
    assign eo_delay[0] = s2_eo_code;

    generate
        for (genvar i = 0; i < EO_DELAY_STAGES; i++) begin : gen_eo_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       eo_delay[i+1] <= bf16_exp2_pkg::EO_PLUS_ONE;
                else if (pipe_en) eo_delay[i+1] <= eo_delay[i];
            end
        end
    endgenerate

    bf16_exp2_pkg::early_out_t eo_aligned;
    assign eo_aligned = eo_delay[EO_DELAY_STAGES];

    // =========================================================================
    // Stage 8: early-out mux and recompose
    //
    // The canonical patterns come from bf16_exp2_pkg on purpose: every core in
    // this repository must emit EXACTLY these so the implementations stay
    // interchangeable over the whole 65536-value input space.
    // =========================================================================
    bf16_exp2_pkg::fp_raw_t s8_final_fp;

    always_comb begin
        unique case (eo_aligned)
            bf16_exp2_pkg::EO_QNAN: begin
                s8_final_fp               = '0;
                s8_final_fp.status.is_nan = 1'b1;
                s8_final_fp.sign          = 1'b1;
                s8_final_fp.mantissa      = 7'b100_0000;
            end
            bf16_exp2_pkg::EO_PLUS_ONE: begin
                s8_final_fp             = '0;
                s8_final_fp.exponent    = 9'sd0;
                s8_final_fp.hidden_bit  = 1'b1;
            end
            bf16_exp2_pkg::EO_PLUS_ZERO: begin
                s8_final_fp                = '0;
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

endmodule : bf16_exp2_optim

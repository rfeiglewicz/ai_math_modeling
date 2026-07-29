// =============================================================================
// bf16_expe_equiv_top.sv
//
// Verification wrapper that instantiates every BF16 exp(x) implementation side
// by side on ONE AXI-Stream input, so the testbench can prove that they are
// drop-in replacements for each other: same latency, same handshake, and
// byte-identical data on every single cycle.
//
// Cores under test (all in base-e mode):
//   0  bf16_exp2         piecewise linear, BRAM coefficients
//   1  bf16_exp2         same, with the DSP shifter + narrowed product
//   2  bf16_expe_lut     full lookup table
//   3  bf16_expe_hybrid  compressed hybrid table
//   4  bf16_expe_cut     cut-point ladder
//   5  bf16_expe_poly4   degree-4 Horner
//   6  bf16_expe_poly4   same, with the DSP front end
//
// bf16_exp2 also implements 2^x, so it carries a tuser bit. It is tied to 0
// (base-e) here because that is the only mode the other cores implement.
//
// Backpressure is shared: all cores see the same m_axis_tready. Because they
// have identical latency they must also assert s_axis_tready and
// m_axis_tvalid on identical cycles, which the testbench checks.
// =============================================================================

module bf16_expe_equiv_top
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b1,
    parameter int PIPE_TARGET     = UNIFIED_PIPE_DEPTH
)(
    input  logic        clk,
    input  logic        rst_n,

    input  logic [15:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,

    input  logic        m_axis_tready,

    // One result bus per core.
    output logic [15:0] o_exp2_tdata,
    output logic        o_exp2_tvalid,
    output logic        o_exp2_tready,

    output logic [15:0] o_exp2opt_tdata,
    output logic        o_exp2opt_tvalid,
    output logic        o_exp2opt_tready,

    output logic [15:0] o_lut_tdata,
    output logic        o_lut_tvalid,
    output logic        o_lut_tready,

    output logic [15:0] o_hybrid_tdata,
    output logic        o_hybrid_tvalid,
    output logic        o_hybrid_tready,

    output logic [15:0] o_cut_tdata,
    output logic        o_cut_tvalid,
    output logic        o_cut_tready,

    output logic [15:0] o_poly4_tdata,
    output logic        o_poly4_tvalid,
    output logic        o_poly4_tready,

    output logic [15:0] o_poly4dsp_tdata,
    output logic        o_poly4dsp_tvalid,
    output logic        o_poly4dsp_tready
);

    // The wrapper's own s_axis_tready is the AND of all cores. If the cores
    // really are equivalent this equals every individual tready.
    assign s_axis_tready = o_exp2_tready & o_exp2opt_tready & o_lut_tready
                         & o_hybrid_tready & o_cut_tready
                         & o_poly4_tready & o_poly4dsp_tready;

    // -------------------------------------------------------------------------
    // 0 - bf16_exp2, baseline
    // -------------------------------------------------------------------------
    bf16_exp2 #(
        .REGISTER_STAGES(REGISTER_STAGES),
        .PIPE_TARGET    (PIPE_TARGET)
    ) u_exp2 (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tuser (1'b0),                 // base-e
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(o_exp2_tready),
        .m_axis_tdata (o_exp2_tdata),
        .m_axis_tvalid(o_exp2_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // -------------------------------------------------------------------------
    // 1 - bf16_exp2, DSP shifter + narrowed log2(e) product + no datapath reset
    // -------------------------------------------------------------------------
    bf16_exp2 #(
        .REGISTER_STAGES     (REGISTER_STAGES),
        .DSP_SHIFT           (1'b1),
        .MANT_MULT_ROUND_FRAC(21),
        .RESET_DATAPATH      (1'b0),
        .PIPE_TARGET         (PIPE_TARGET)
    ) u_exp2_opt (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tuser (1'b0),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(o_exp2opt_tready),
        .m_axis_tdata (o_exp2opt_tdata),
        .m_axis_tvalid(o_exp2opt_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // -------------------------------------------------------------------------
    // 2 - full lookup table
    // -------------------------------------------------------------------------
    bf16_expe_lut #(
        .REGISTER_STAGES(REGISTER_STAGES),
        .PIPE_TARGET    (PIPE_TARGET)
    ) u_lut (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(o_lut_tready),
        .m_axis_tdata (o_lut_tdata),
        .m_axis_tvalid(o_lut_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // -------------------------------------------------------------------------
    // 3 - hybrid compressed table
    // -------------------------------------------------------------------------
    bf16_expe_hybrid #(
        .REGISTER_STAGES(REGISTER_STAGES),
        .PIPE_TARGET    (PIPE_TARGET)
    ) u_hybrid (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(o_hybrid_tready),
        .m_axis_tdata (o_hybrid_tdata),
        .m_axis_tvalid(o_hybrid_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // -------------------------------------------------------------------------
    // 4 - cut-point ladder
    // -------------------------------------------------------------------------
    bf16_expe_cut #(
        .REGISTER_STAGES(REGISTER_STAGES),
        .PIPE_TARGET    (PIPE_TARGET)
    ) u_cut (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(o_cut_tready),
        .m_axis_tdata (o_cut_tdata),
        .m_axis_tvalid(o_cut_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // -------------------------------------------------------------------------
    // 5 - degree-4 Horner, fabric front end
    // -------------------------------------------------------------------------
    bf16_expe_poly4 #(
        .REGISTER_STAGES(REGISTER_STAGES),
        .PIPE_TARGET    (PIPE_TARGET)
    ) u_poly4 (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(o_poly4_tready),
        .m_axis_tdata (o_poly4_tdata),
        .m_axis_tvalid(o_poly4_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // -------------------------------------------------------------------------
    // 6 - degree-4 Horner, DSP front end
    // -------------------------------------------------------------------------
    bf16_expe_poly4 #(
        .REGISTER_STAGES(REGISTER_STAGES),
        .DSP_FRONTEND   (1'b1),
        .RESET_DATAPATH (1'b0),
        .PIPE_TARGET    (PIPE_TARGET)
    ) u_poly4_dsp (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata (s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(o_poly4dsp_tready),
        .m_axis_tdata (o_poly4dsp_tdata),
        .m_axis_tvalid(o_poly4dsp_tvalid),
        .m_axis_tready(m_axis_tready)
    );

endmodule : bf16_expe_equiv_top

// =============================================================================
// bf16_expe_hybrid.sv
// Hybrid compressed-table BF16 exp(x), AXI-Stream interface.
//
// Core representation (handled range [-9, 6]):
//   exponent [-9,-2] -> threshold decoder (98 x 7-bit thresholds + 8 bases)
//   exponent [-1, 6] -> dense 1024 x 16-bit ROM
// Exponent 7 (x <= -128, always +0.0) is answered by bf16_early_out with
// MAX_EXP = 6, so the datapath needs no zero route: with idx = exp + 9 in
// [0, 15], idx[3] selects dense vs sparse and idx[2:0] is the 3-bit index of
// both halves (exp + 9 and exp + 1 agree modulo 8).
// Results are identical to bf16_expe_lut_approx() for all 2^16 inputs.
// =============================================================================

module bf16_expe_hybrid
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b0,
    // Pad the output to this many pipeline stages so every core has the same
    // latency. 0 = natural depth. See bf16_exp2_pkg::UNIFIED_PIPE_DEPTH.
    parameter int PIPE_TARGET     = UNIFIED_PIPE_DEPTH,
    // -GREPORT_DEPTH=1 prints the resulting pipeline depth at elaboration.
    parameter bit REPORT_DEPTH    = 1'b0
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    output logic [15:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready
);
    localparam int CORE_DEPTH = REGISTER_STAGES ? 4 : 0;
    localparam int PAD_STAGES = (REGISTER_STAGES && PIPE_TARGET > CORE_DEPTH)
                                ? PIPE_TARGET - CORE_DEPTH : 0;
    localparam int PIPE_DEPTH = CORE_DEPTH + PAD_STAGES;

    // See bf16_expe_lut.sv for why this exists.
    if (REPORT_DEPTH) begin : gen_depth_report
        $info("CORE_PIPE_DEPTH core=%0d pad=%0d total=%0d",
              CORE_DEPTH, PAD_STAGES, PIPE_DEPTH);
    end

    localparam int MAX_EXP = 6;   // last exponent handled by the tables

    logic pipe_en;

    generate
        if (REGISTER_STAGES) begin : gen_axi_pipelined
            logic [PIPE_DEPTH-1:0] valid_sr;
            assign m_axis_tvalid = valid_sr[PIPE_DEPTH-1];
            assign pipe_en       = m_axis_tready | ~m_axis_tvalid;
            assign s_axis_tready = pipe_en;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) valid_sr <= '0;
                else if (pipe_en) valid_sr <= {valid_sr[PIPE_DEPTH-2:0], s_axis_tvalid};
            end
        end else begin : gen_axi_comb
            assign m_axis_tvalid = s_axis_tvalid;
            assign s_axis_tready = m_axis_tready;
            assign pipe_en       = 1'b1;
        end
    endgenerate

    // Stage 1: BF16 decomposition.
    fp_raw_t s1_decomposed;
    bf16_decompose #(.REGISTER_OUTPUT(REGISTER_STAGES)) u_decompose (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .bf16_in(s_axis_tdata), .decomposed(s1_decomposed)
    );

    // Stage 2: early out, path selection, and compact addresses.
    early_out_t s2_eo_code;
    bf16_early_out #(
        .REGISTER_OUTPUT(REGISTER_STAGES),
        .MAX_EXP        (MAX_EXP)
    ) u_early_out (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .decomposed(s1_decomposed), .eo_code(s2_eo_code)
    );

    // idx = exp + 9 in [0, 15] whenever eo_code == EO_NONE.
    //   idx[3] = 0 -> exp in [-9, -2] -> sparse decoder, bin idx[2:0]
    //   idx[3] = 1 -> exp in [-1,  6] -> dense ROM,      bin idx[2:0]
    logic signed [8:0] exp_offset_comb;
    logic [3:0] exp_idx_comb;
    logic route_dense_comb;
    logic [2:0] bin_index_comb;
    logic [9:0] dense_addr_comb;

    assign exp_offset_comb  = s1_decomposed.exponent + 9'sd9;
    assign exp_idx_comb     = exp_offset_comb[3:0];
    assign route_dense_comb = exp_idx_comb[3];
    assign bin_index_comb   = exp_idx_comb[2:0];
    assign dense_addr_comb  = {bin_index_comb, s1_decomposed.mantissa};

    logic s2_route_dense;
    logic [2:0] s2_sparse_exp_index;
    logic [6:0] s2_mantissa;
    logic [9:0] s2_dense_addr;

    generate
        if (REGISTER_STAGES) begin : gen_stage2_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s2_route_dense      <= 1'b0;
                    s2_sparse_exp_index <= '0;
                    s2_mantissa         <= '0;
                    s2_dense_addr       <= '0;
                end else if (pipe_en) begin
                    s2_route_dense      <= route_dense_comb;
                    s2_sparse_exp_index <= bin_index_comb;
                    s2_mantissa         <= s1_decomposed.mantissa;
                    s2_dense_addr       <= dense_addr_comb;
                end
            end
        end else begin : gen_stage2_wires
            assign s2_route_dense      = route_dense_comb;
            assign s2_sparse_exp_index = bin_index_comb;
            assign s2_mantissa         = s1_decomposed.mantissa;
            assign s2_dense_addr       = dense_addr_comb;
        end
    endgenerate

    // Stage 3: sparse threshold decoder and dense ROM operate in parallel.
    logic [15:0] s3_sparse_data;
    logic [15:0] s3_dense_data;

    bf16_expe_sparse_decode #(.REGISTERED(REGISTER_STAGES)) u_sparse (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .exp_index(s2_sparse_exp_index), .mantissa(s2_mantissa),
        .data(s3_sparse_data)
    );

    bf16_expe_hybrid_rom #(.REGISTERED(REGISTER_STAGES)) u_dense_rom (
        .clk(clk), .pipe_en(pipe_en), .addr(s2_dense_addr), .data(s3_dense_data)
    );

    early_out_t s3_eo_code;
    logic s3_route_dense;

    generate
        if (REGISTER_STAGES) begin : gen_stage3_align
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s3_eo_code     <= EO_PLUS_ONE;
                    s3_route_dense <= 1'b0;
                end else if (pipe_en) begin
                    s3_eo_code     <= s2_eo_code;
                    s3_route_dense <= s2_route_dense;
                end
            end
        end else begin : gen_stage3_align_wires
            assign s3_eo_code     = s2_eo_code;
            assign s3_route_dense = s2_route_dense;
        end
    endgenerate

    // Stage 4: early-out and route mux.
    logic [15:0] result_comb;
    always_comb begin
        unique case (s3_eo_code)
            EO_QNAN:      result_comb = BF16_QNAN;
            EO_PLUS_ONE:  result_comb = BF16_PLUS_ONE;
            EO_PLUS_ZERO: result_comb = BF16_PLUS_ZERO;
            default:      result_comb = s3_route_dense ? s3_dense_data
                                                       : s3_sparse_data;
        endcase
    end

    logic [15:0] core_data;

    generate
        if (REGISTER_STAGES) begin : gen_output_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) core_data <= 16'h0000;
                else if (pipe_en) core_data <= result_comb;
            end
        end else begin : gen_output_wire
            assign core_data = result_comb;
        end
    endgenerate

    // Latency padding so this core matches the deepest implementation.
    bf16_pipe_pad #(
        .STAGES(PAD_STAGES),
        .WIDTH (16)
    ) u_pipe_pad (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .d(core_data), .q(m_axis_tdata)
    );
endmodule : bf16_expe_hybrid

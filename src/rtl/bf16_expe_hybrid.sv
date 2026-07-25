// =============================================================================
// bf16_expe_hybrid.sv
// Hybrid compressed-table BF16 exp(x), AXI-Stream interface.
//
// Core representation:
//   exponent [-9,-2] -> threshold decoder (98 x 7-bit thresholds + 8 bases)
//   exponent [-1, 6] -> dense 1024 x 16-bit ROM
//   exponent 7       -> constant +0.0
// Early-out behavior is identical to bf16_expe_lut_approx().
// =============================================================================

module bf16_expe_hybrid
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b0
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
    localparam int PIPE_DEPTH = REGISTER_STAGES ? 4 : 0;

    typedef enum logic [1:0] {
        ROUTE_SPARSE = 2'b00,
        ROUTE_DENSE  = 2'b01,
        ROUTE_ZERO   = 2'b10
    } route_t;

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
    bf16_early_out #(.REGISTER_OUTPUT(REGISTER_STAGES)) u_early_out (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .decomposed(s1_decomposed), .eo_code(s2_eo_code)
    );

    route_t route_comb;
    logic signed [8:0] sparse_offset_comb;
    logic signed [8:0] dense_offset_comb;
    logic [2:0] sparse_exp_index_comb;
    logic [2:0] dense_exp_index_comb;
    logic [9:0] dense_addr_comb;

    always_comb begin
        if ($signed(s1_decomposed.exponent) <= -9'sd2)
            route_comb = ROUTE_SPARSE;
        else if ($signed(s1_decomposed.exponent) <= 9'sd6)
            route_comb = ROUTE_DENSE;
        else
            route_comb = ROUTE_ZERO;
    end

    assign sparse_offset_comb    = s1_decomposed.exponent + 9'sd9;
    assign dense_offset_comb     = s1_decomposed.exponent + 9'sd1;
    assign sparse_exp_index_comb = sparse_offset_comb[2:0];
    assign dense_exp_index_comb  = dense_offset_comb[2:0];
    assign dense_addr_comb       = {dense_exp_index_comb, s1_decomposed.mantissa};

    route_t s2_route;
    logic [2:0] s2_sparse_exp_index;
    logic [6:0] s2_mantissa;
    logic [9:0] s2_dense_addr;

    generate
        if (REGISTER_STAGES) begin : gen_stage2_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s2_route            <= ROUTE_SPARSE;
                    s2_sparse_exp_index <= '0;
                    s2_mantissa         <= '0;
                    s2_dense_addr       <= '0;
                end else if (pipe_en) begin
                    s2_route            <= route_comb;
                    s2_sparse_exp_index <= sparse_exp_index_comb;
                    s2_mantissa         <= s1_decomposed.mantissa;
                    s2_dense_addr       <= dense_addr_comb;
                end
            end
        end else begin : gen_stage2_wires
            assign s2_route            = route_comb;
            assign s2_sparse_exp_index = sparse_exp_index_comb;
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
    route_t s3_route;

    generate
        if (REGISTER_STAGES) begin : gen_stage3_align
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s3_eo_code <= EO_PLUS_ONE;
                    s3_route   <= ROUTE_SPARSE;
                end else if (pipe_en) begin
                    s3_eo_code <= s2_eo_code;
                    s3_route   <= s2_route;
                end
            end
        end else begin : gen_stage3_align_wires
            assign s3_eo_code = s2_eo_code;
            assign s3_route   = s2_route;
        end
    endgenerate

    // Stage 4: early-out and route mux.
    logic [15:0] result_comb;
    always_comb begin
        unique case (s3_eo_code)
            EO_QNAN:      result_comb = 16'hFFC0;
            EO_PLUS_ONE:  result_comb = 16'h3F80;
            EO_PLUS_ZERO: result_comb = 16'h0000;
            default: begin
                unique case (s3_route)
                    ROUTE_SPARSE: result_comb = s3_sparse_data;
                    ROUTE_DENSE:  result_comb = s3_dense_data;
                    default:      result_comb = 16'h0000;
                endcase
            end
        endcase
    end

    generate
        if (REGISTER_STAGES) begin : gen_output_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) m_axis_tdata <= 16'h0000;
                else if (pipe_en) m_axis_tdata <= result_comb;
            end
        end else begin : gen_output_wire
            assign m_axis_tdata = result_comb;
        end
    endgenerate
endmodule : bf16_expe_hybrid

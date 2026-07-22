// =============================================================================
// bf16_expe_lut.sv
// Top-level module for the BF16 exp(x) (base-e only) FULL LOOKUP-TABLE model.
// AXI-Stream interface: s_axis (slave/input) and m_axis (master/output).
//
// This is the hardware counterpart of the C++ model bf16_expe_lut_approx()
// (src/approximations/bf16_expe_lut.hpp). Instead of a log2(e) multiply plus a
// piecewise-linear approximation, every negative input whose unbiased exponent
// lies in [-9, 7] is resolved by a direct, correctly-rounded (RNE) table lookup.
// The result is therefore <= 0.5 ULP by construction over the tabulated range.
//
// Architecture (3 functional sub-modules):
//   1. bf16_decompose     - BF16 bit field decomposition
//   2. bf16_early_out     - Special case / range check detector
//   3. bf16_expe_lut_rom  - 2176 x 16-bit correctly-rounded exp(x) ROM (BRAM)
//
// Output selection:
//   EO_QNAN      -> 0xFFC0  (qNaN indefinite)
//   EO_PLUS_ONE  -> 0x3F80  (+1.0)
//   EO_PLUS_ZERO -> 0x0000  (+0.0)
//   EO_NONE      -> ROM data (correctly-rounded exp(x))
//
// Pipeline notes:
//   When REGISTER_STAGES=1, four register stages are inserted:
//     T+1 decompose | T+2 early_out + ROM address | T+3 ROM read | T+4 output
//   pipe_en gates ALL pipeline registers for AXI-Stream backpressure.
//
// Pipeline depth (REGISTER_STAGES=1): 4 FF stages -> 4 cycle latency.
// Pipeline depth (REGISTER_STAGES=0): 0 (fully combinational, single-cycle).
// =============================================================================

module bf16_expe_lut
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b0   // Enable per-stage registers
)(
    input  logic        clk,
    input  logic        rst_n,

    // Slave AXI-Stream (input)
    input  logic [15:0] s_axis_tdata,    // BF16 input value
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,

    // Master AXI-Stream (output)
    output logic [15:0] m_axis_tdata,    // BF16 result = exp(input)
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready
);

    // =========================================================================
    // Table geometry (mirrors bf16_expe_lut_table.hpp)
    // =========================================================================
    localparam int LUT_MANT_COUNT = 128;                       // 2^7 mantissa steps
    localparam int LUT_DEPTH       = 2176;                      // 17 exponents * 128
    localparam int LUT_ADDR_WIDTH  = 12;                        // $clog2(2176)
    localparam int EXP_IDX_W        = 5;                        // (exp+9) in [0,16]

    // Early-out constant BF16 payloads
    localparam logic [15:0] BF16_QNAN      = 16'hFFC0;
    localparam logic [15:0] BF16_PLUS_ONE  = 16'h3F80;
    localparam logic [15:0] BF16_PLUS_ZERO = 16'h0000;

    // =========================================================================
    // Pipeline depth: 4 register stages when REGISTER_STAGES=1.
    //   decompose | early_out+addr | ROM read | output mux
    // =========================================================================
    localparam int PIPE_DEPTH = REGISTER_STAGES ? 4 : 0;

    // =========================================================================
    // AXI-Stream handshake & pipeline enable ("all-stall" approach)
    //   pipe_en=1 => pipeline advances,  pipe_en=0 => pipeline holds
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

    // =========================================================================
    // Stage 1: Decompose
    // =========================================================================
    fp_raw_t s1_decomposed;

    bf16_decompose #(
        .REGISTER_OUTPUT(REGISTER_STAGES)
    ) u_decompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .pipe_en   (pipe_en),
        .bf16_in   (s_axis_tdata),
        .decomposed(s1_decomposed)
    );

    // =========================================================================
    // Stage 2: Early-out detection + ROM address computation
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

    // ROM address = (exponent - LUT_MIN_EXP) * 128 + mantissa
    //             = (exponent + 9) << 7 | mantissa
    // Only meaningful when eo_code == EO_NONE (negative, exp in [-9, 7]).
    logic signed [8:0]         exp_offset_comb;
    logic [EXP_IDX_W-1:0]      exp_index_comb;
    logic [LUT_ADDR_WIDTH-1:0] addr_comb;

    assign exp_offset_comb = s1_decomposed.exponent - 9'(INPUT_MIN_EXP);
    assign exp_index_comb  = exp_offset_comb[EXP_IDX_W-1:0];
    assign addr_comb       = {exp_index_comb, s1_decomposed.mantissa};

    // Register the address so it is aligned with s2_eo_code (both at T+2).
    logic [LUT_ADDR_WIDTH-1:0] s2_addr;

    generate
        if (REGISTER_STAGES) begin : gen_addr_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       s2_addr <= '0;
                else if (pipe_en) s2_addr <= addr_comb;
            end
        end else begin : gen_addr_wire
            assign s2_addr = addr_comb;
        end
    endgenerate

    // =========================================================================
    // Stage 3: ROM read (correctly-rounded exp(x) BF16 output)
    //   REGISTERED=1 -> data available one cycle after addr (T+3).
    // =========================================================================
    logic [15:0] s3_rom_data;

    bf16_expe_lut_rom #(
        .DATA_W    (16),
        .DEPTH     (LUT_DEPTH),
        .ADDR_W    (LUT_ADDR_WIDTH),
        .REGISTERED(REGISTER_STAGES)
    ) u_rom (
        .clk    (clk),
        .pipe_en(pipe_en),
        .addr   (s2_addr),
        .data   (s3_rom_data)
    );

    // =========================================================================
    // Delay chain for eo_code to align with s3_rom_data.
    //   REGISTER_STAGES=1: eo at T+2, ROM data at T+3 -> 1 extra FF.
    // =========================================================================
    localparam int EO_DELAY_STAGES = REGISTER_STAGES ? 1 : 0;

    early_out_t eo_delay [EO_DELAY_STAGES+1];
    assign eo_delay[0] = s2_eo_code;

    generate
        for (genvar i = 0; i < EO_DELAY_STAGES; i++) begin : gen_eo_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       eo_delay[i+1] <= EO_PLUS_ONE;
                else if (pipe_en) eo_delay[i+1] <= eo_delay[i];
            end
        end
    endgenerate

    early_out_t s3_eo_aligned;
    assign s3_eo_aligned = eo_delay[EO_DELAY_STAGES];

    // =========================================================================
    // Stage 4: Output mux (early-out constants vs ROM data) + output register
    // =========================================================================
    logic [15:0] bf16_out_comb;

    always_comb begin
        unique case (s3_eo_aligned)
            EO_QNAN:      bf16_out_comb = BF16_QNAN;
            EO_PLUS_ONE:  bf16_out_comb = BF16_PLUS_ONE;
            EO_PLUS_ZERO: bf16_out_comb = BF16_PLUS_ZERO;
            default:      bf16_out_comb = s3_rom_data;   // EO_NONE
        endcase
    end

    generate
        if (REGISTER_STAGES) begin : gen_out_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       m_axis_tdata <= 16'h0000;
                else if (pipe_en) m_axis_tdata <= bf16_out_comb;
            end
        end else begin : gen_out_wire
            assign m_axis_tdata = bf16_out_comb;
        end
    endgenerate

endmodule : bf16_expe_lut

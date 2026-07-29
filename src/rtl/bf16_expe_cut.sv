// =============================================================================
// bf16_expe_cut.sv
// Cut-point ladder BF16 exp(x), AXI-Stream interface.
//
// Core representation:
//     exp(x) = 2^(-t),  t = -x*log2(e),  I = floor(t),  f = frac(t)
//            = 2^(-I) * 2^(-f)
//
// The output exponent is -I-1 (pure integer arithmetic) and the output mantissa
// depends ONLY on f, so all 17 input exponent bins share ONE ladder -- the
// tables no longer scale with the exponent range.
//
// Datapath (one constant multiply, one shift, one comparison):
//     prod  = (1.mantissa) * LOG2E_Q                 8 x 24 constant multiply
//     algn  = prod >> (LOG2E_FRAC_BITS + 7 - exp - FRAC_BITS)
//     I     = algn[ALIGNED_W-1:FRAC_BITS]            output exponent = -I-1
//     f     = algn[FRAC_BITS-1:0]
//     base  = (128 - (f[15:8] >> 1)) + CAND_DEV_BIAS + cand_rom[f[15:8]]
//     cut   = ((128-k) << 9) + CUT_DEV_BIAS + cut_rom[k-1],   k = base+1
//     mant  = base + (f <= cut)                      SINGLE comparison
//
// The stored cut points are not the exact mantissa breakpoints; they only
// separate the f values that can actually occur, which is what lets each entry
// fit in CUT_DEV_BITS instead of FRAC_BITS.
//
// Early-out behavior is identical to bf16_expe_lut_approx() and
// bf16_expe_hybrid_approx().
// =============================================================================

module bf16_expe_cut
    import bf16_exp2_pkg::*;
    import bf16_expe_cut_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b0,
    // Pad the output to this many pipeline stages so every core has the same
    // latency. 0 = natural depth. See bf16_exp2_pkg::UNIFIED_PIPE_DEPTH.
    parameter int PIPE_TARGET     = UNIFIED_PIPE_DEPTH
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

    typedef enum logic [1:0] {
        ROUTE_LADDER = 2'b00,
        ROUTE_TAIL   = 2'b01,
        ROUTE_ZERO   = 2'b10
    } route_t;

    logic pipe_en;

    // -------------------------------------------------------------------------
    // AXI-Stream handshake
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Stage 1: BF16 decomposition
    // -------------------------------------------------------------------------
    fp_raw_t s1_decomposed;
    bf16_decompose #(.REGISTER_OUTPUT(REGISTER_STAGES)) u_decompose (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .bf16_in(s_axis_tdata), .decomposed(s1_decomposed)
    );

    // -------------------------------------------------------------------------
    // Stage 2: early out, routing, constant multiply and alignment shift
    // -------------------------------------------------------------------------
    early_out_t s2_eo_code;
    bf16_early_out #(.REGISTER_OUTPUT(REGISTER_STAGES)) u_early_out (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .decomposed(s1_decomposed), .eo_code(s2_eo_code)
    );

    // Routing inside the approximated range: the ladder covers everything up to
    // the point where exp(x) stops being a normal BF16 number.
    route_t route_comb;
    always_comb begin
        if ($signed(s1_decomposed.exponent) > $signed(9'(ZERO_EXP)))
            route_comb = ROUTE_ZERO;
        else if ($signed(s1_decomposed.exponent) == $signed(9'(ZERO_EXP))
                 && s1_decomposed.mantissa >= 7'(ZERO_MANT_LO))
            route_comb = ROUTE_ZERO;
        else if ($signed(s1_decomposed.exponent) == $signed(9'(TAIL_EXP))
                 && s1_decomposed.mantissa >= 7'(TAIL_MANT_LO))
            route_comb = ROUTE_TAIL;   // mantissa <= TAIL_MANT_HI implied above
        else
            route_comb = ROUTE_LADDER;
    end

    // Constant multiply: (1.mantissa) * log2(e).  One operand is a constant, so
    // synthesis maps this to a small CSD adder tree rather than a DSP block.
    logic [MANT_SRC_W-1:0]  mant_ext_comb;
    logic [PROD_W-1:0]      prod_comb;
    assign mant_ext_comb = {1'b1, s1_decomposed.mantissa};
    assign prod_comb     = mant_ext_comb * LOG2E_Q;

    // Single variable shift places the binary point: FRAC_BITS fractional bits.
    logic signed [8:0]      shift_signed;
    logic [4:0]             shift_comb;
    logic [ALIGNED_W-1:0]   aligned_comb;
    assign shift_signed = $signed(9'(LOG2E_FRAC_BITS + 7 - FRAC_BITS))
                        - $signed(s1_decomposed.exponent);
    assign shift_comb   = shift_signed[4:0];   // in [7, 23] for exp in [-9, 7]
    assign aligned_comb = ALIGNED_W'(prod_comb >> shift_comb);

    logic [INT_W-1:0]  int_part_comb;
    logic [FRAC_BITS-1:0] frac_comb;
    assign int_part_comb = aligned_comb[ALIGNED_W-1:FRAC_BITS];
    assign frac_comb     = aligned_comb[FRAC_BITS-1:0];

    logic [3:0] tail_addr_comb;
    assign tail_addr_comb = 4'(s1_decomposed.mantissa - 7'(TAIL_MANT_LO));

    route_t               s2_route;
    logic [INT_W-1:0]     s2_int_part;
    logic [FRAC_BITS-1:0] s2_frac;
    logic [3:0]           s2_tail_addr;

    generate
        if (REGISTER_STAGES) begin : gen_stage2_regs
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s2_route     <= ROUTE_ZERO;
                    s2_int_part  <= '0;
                    s2_frac      <= '0;
                    s2_tail_addr <= '0;
                end else if (pipe_en) begin
                    s2_route     <= route_comb;
                    s2_int_part  <= int_part_comb;
                    s2_frac      <= frac_comb;
                    s2_tail_addr <= tail_addr_comb;
                end
            end
        end else begin : gen_stage2_wires
            assign s2_route     = route_comb;
            assign s2_int_part  = int_part_comb;
            assign s2_frac      = frac_comb;
            assign s2_tail_addr = tail_addr_comb;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Stage 3: candidate ROM and subnormal tail ROM read in parallel
    // -------------------------------------------------------------------------
    logic [CAND_DEV_BITS-1:0] s3_cand_dev;
    logic [6:0]               s3_tail_data;

    bf16_expe_cut_cand_rom #(.REGISTERED(REGISTER_STAGES)) u_cand_rom (
        .clk(clk), .pipe_en(pipe_en),
        .addr(s2_frac[FRAC_BITS-1 -: CAND_INDEX_BITS]),
        .data(s3_cand_dev)
    );

    bf16_expe_cut_tail_rom #(.REGISTERED(REGISTER_STAGES)) u_tail_rom (
        .clk(clk), .pipe_en(pipe_en),
        .addr(s2_tail_addr), .data(s3_tail_data)
    );

    early_out_t           s3_eo_code;
    route_t               s3_route;
    logic [INT_W-1:0]     s3_int_part;
    logic [FRAC_BITS-1:0] s3_frac;

    generate
        if (REGISTER_STAGES) begin : gen_stage3_align
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s3_eo_code  <= EO_PLUS_ONE;
                    s3_route    <= ROUTE_ZERO;
                    s3_int_part <= '0;
                    s3_frac     <= '0;
                end else if (pipe_en) begin
                    s3_eo_code  <= s2_eo_code;
                    s3_route    <= s2_route;
                    s3_int_part <= s2_int_part;
                    s3_frac     <= s2_frac;
                end
            end
        end else begin : gen_stage3_align_wires
            assign s3_eo_code  = s2_eo_code;
            assign s3_route    = s2_route;
            assign s3_int_part = s2_int_part;
            assign s3_frac     = s2_frac;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Stage 4: candidate reconstruction, one cut comparison, pack
    // -------------------------------------------------------------------------
    // base = 128 - (index >> (CAND_INDEX_BITS-7)) + CAND_DEV_BIAS + cand_dev
    logic [CAND_INDEX_BITS-1:0] cand_index;
    logic signed [8:0]          base_count;
    assign cand_index = s3_frac[FRAC_BITS-1 -: CAND_INDEX_BITS];
    assign base_count = 9'sd128
                      - $signed({2'b00, cand_index[CAND_INDEX_BITS-1:CAND_INDEX_BITS-7]})
                      + $signed(9'(CAND_DEV_BIAS))
                      + $signed({5'b00000, s3_cand_dev});

    // Cut point k = base+1.  k == 129 is the sentinel "nothing compares below",
    // which happens when the candidate already saturates the mantissa at 128.
    logic [6:0]                cut_addr;
    logic [CUT_DEV_BITS-1:0]   cut_dev;
    logic signed [CUT_W-1:0]   cut_value;
    logic                      cut_sentinel;

    assign cut_addr     = base_count[6:0];             // (k-1) = base
    assign cut_sentinel = (base_count >= 9'sd128);

    bf16_expe_cut_cut_rom #(.REGISTERED(1'b0)) u_cut_rom (
        .clk(clk), .pipe_en(pipe_en), .addr(cut_addr), .data(cut_dev)
    );

    // cut = ((128-k) << (FRAC_BITS-7)) + CUT_DEV_BIAS + cut_dev,  128-k = 127-base
    logic signed [CUT_W-1:0] cut_linear;
    assign cut_linear = $signed(CUT_W'($signed(9'sd127 - base_count))) <<< (FRAC_BITS - 7);
    assign cut_value  = cut_linear + $signed(CUT_W'(CUT_DEV_BIAS))
                      + $signed({{(CUT_W-CUT_DEV_BITS){1'b0}}, cut_dev});

    logic                below_cut;
    logic signed [8:0]   ladder_count;
    assign below_cut    = !cut_sentinel
                        && ($signed({{(CUT_W-FRAC_BITS){1'b0}}, s3_frac}) <= cut_value);
    assign ladder_count = base_count + 9'(below_cut);

    // exponent = -I-1, biased = 127 + (-I-1) = 126 - I.
    // count == 128 means the mantissa rounded up to 2.0: mantissa 0, exponent +1.
    // count[6:0] is already 0 in that case, so only the exponent needs the carry.
    logic [7:0] biased_exp;
    logic [6:0] out_mantissa;
    assign biased_exp   = 8'd126 - 8'(s3_int_part) + 8'(ladder_count[7]);
    assign out_mantissa = ladder_count[6:0];

    logic [15:0] ladder_result;
    assign ladder_result = {1'b0, biased_exp, out_mantissa};

    // -------------------------------------------------------------------------
    // Early-out and route mux
    // -------------------------------------------------------------------------
    logic [15:0] result_comb;
    always_comb begin
        unique case (s3_eo_code)
            EO_QNAN:      result_comb = BF16_QNAN;
            EO_PLUS_ONE:  result_comb = BF16_PLUS_ONE;
            EO_PLUS_ZERO: result_comb = BF16_PLUS_ZERO;
            default: begin
                unique case (s3_route)
                    ROUTE_LADDER: result_comb = ladder_result;
                    ROUTE_TAIL:   result_comb = {9'b0, s3_tail_data};
                    default:      result_comb = 16'h0000;
                endcase
            end
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
endmodule : bf16_expe_cut

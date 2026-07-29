// =============================================================================
// bf16_expe_poly4.sv
// Degree-4 minimax/Horner BF16 exp(x), AXI-Stream interface.
//
// Core representation (identical to bf16_expe_cut):
//     exp(x) = 2^(-t),  t = -x*log2(e),  I = floor(t),  f = frac(t)
//            = 2^(-I-1) * 2^(1-f)
//
// The output exponent is -I-1 (pure integer arithmetic) and the output mantissa
// depends ONLY on f.  Where bf16_expe_cut classifies f against stored cut
// points, this variant evaluates a degree-4 polynomial P(f) ~= 2^(1-f) and
// rounds.  It is the DSP-heavy / storage-light end of the same design space:
//
//     bf16_expe_cut   : 2765 table bits, 0 variable multipliers
//     bf16_expe_poly4 :  157 table bits, 4 variable multipliers
//
// DSP mapping
// -----------
// A DSP48E1 computes P = A*B + C with A 25-bit signed and B 18-bit signed.
// With ACC_W = 25 and FRAC_BITS = 17 one Horner step is exactly one slice:
//     A <- acc, B <- f, C <- coefficient pre-shifted into the product scale
// The round-to-nearest constant is folded into C, so rounding at every Horner
// step costs nothing.  The four steps chain through PCIN as a DSP cascade.
//
// Datapath
//     prod  = (1.mantissa) * LOG2E_Q                8 x 24 CONSTANT multiply
//     algn  = prod >> (LOG2E_FRAC_BITS + 7 - exp - FRAC_BITS)
//     I     = algn[ALIGNED_W-1:FRAC_BITS]           output exponent = -I-1
//     f     = algn[FRAC_BITS-1:0]
//     acc   = C4
//     acc   = (acc*f + C_TERM[k]) >>> FRAC_BITS     k = 3,2,1,0   -> 4 DSPs
//     mant  = (acc - ACC_ONE + ACC_HALF) >>> ROUND_SHIFT
//
// The coefficients are not minimax coefficients: the generator maximizes the
// worst-case distance to a rounding boundary on this exact integer datapath,
// so coefficient quantization and per-step truncation are inside the objective
// rather than added on top of it.  See gen_bf16_expe_poly4_tables.cpp.
//
// Early-out behavior is identical to bf16_expe_lut_approx(),
// bf16_expe_hybrid_approx() and bf16_expe_cut_approx().
// =============================================================================

// One Horner step: acc <- (acc*f + c_term) >>> FRAC_BITS.
// Kept as a separate module so synthesis sees a clean A*B + C pattern.
/* verilator lint_off DECLFILENAME */
module bf16_expe_poly4_step
    import bf16_expe_poly4_pkg::*;
#(
    parameter bit REGISTERED     = 1'b1,
    parameter bit RESET_DATAPATH = 1'b0
)(
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                      clk,
    input  logic                      rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                      pipe_en,
    input  logic signed [ACC_W-1:0]   acc_in,
    input  logic [FRAC_BITS-1:0]      frac,
    input  logic signed [PROD_W-1:0]  c_term,
    output logic signed [ACC_W-1:0]   acc_out
);
    logic signed [PROD_W-1:0] prod;
    logic signed [ACC_W-1:0]  next;

    // A * B + C.  frac is unsigned, so it is zero-extended by one bit before
    // being treated as a signed DSP operand.
    assign prod = acc_in * $signed({1'b0, frac}) + c_term;

    // Arithmetic shift: floor division, matching the C++ model exactly.
    assign next = ACC_W'(prod >>> FRAC_BITS);

    generate
        if (REGISTERED && RESET_DATAPATH) begin : gen_reg_rst
            // An asynchronous reset cannot live in the DSP: DSP48E1 only has a
            // synchronous reset on PREG.  Vivado keeps PREG anyway and gates
            // the D input instead, which costs one LUT per accumulator bit.
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) acc_out <= '0;
                else if (pipe_en) acc_out <= next;
            end
        end else if (REGISTERED) begin : gen_reg
            // No reset: the register maps straight onto PREG, zero fabric.
            // Stale data is harmless because m_axis_tvalid comes from the
            // valid shift register, which is reset.
            always_ff @(posedge clk) begin
                if (pipe_en) acc_out <= next;
            end
        end else begin : gen_wire
            assign acc_out = next;
        end
    endgenerate
endmodule : bf16_expe_poly4_step
/* verilator lint_on DECLFILENAME */


module bf16_expe_poly4
    import bf16_exp2_pkg::*;
    import bf16_expe_poly4_pkg::*;
#(
    parameter bit REGISTER_STAGES = 1'b0,
    // 0: alignment uses a CSD adder tree and a barrel shifter (fabric only).
    // 1: alignment uses three more DSP slices and almost no fabric.  Both are
    //    bit-identical; see the front-end block below.
    parameter bit DSP_FRONTEND     = 1'b0,
    // Asynchronous reset on the datapath pipeline registers.  Leave this off
    // unless the surrounding system really needs deterministic data during
    // reset: DSP48E1 has no asynchronous reset, so turning it on forces Vivado
    // to gate every DSP input in fabric, at roughly one LUT per bit.  Control
    // state (valid shift register, output register) is always reset.
    parameter bit RESET_DATAPATH   = 1'b0,
    // Pad the output to this many pipeline stages so every core has the same
    // latency. 0 = natural depth. See bf16_exp2_pkg::UNIFIED_PIPE_DEPTH.
    parameter int PIPE_TARGET      = UNIFIED_PIPE_DEPTH
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
    // The DSP front-end needs one extra register: the one-hot scaling and the
    // wide multiply cannot share a pipeline stage without hurting Fmax.
    localparam int FE_EXTRA = (DSP_FRONTEND && REGISTER_STAGES) ? 1 : 0;

    // decompose + align + DEGREE Horner steps + output register
    localparam int CORE_DEPTH = REGISTER_STAGES ? (3 + FE_EXTRA + DEGREE) : 0;
    localparam int PAD_STAGES = (REGISTER_STAGES && PIPE_TARGET > CORE_DEPTH)
                                ? PIPE_TARGET - CORE_DEPTH : 0;
    localparam int PIPE_DEPTH = CORE_DEPTH + PAD_STAGES;

    // Coefficient shifted from COEF_FRAC scale into the product scale, with the
    // rounding constant folded in.  This is the DSP C-port value.
    localparam int C_SHIFT = ACC_FRAC - COEF_FRAC + FRAC_BITS;

    localparam int ACC_ONE_I  = 1 << ACC_FRAC;
    localparam int ACC_HALF_I = 1 << (ACC_FRAC - 8);

    // The "- 1.0 + half an output LSB" of the final rounding step is a constant,
    // so it rides along in the C port of the last Horner DSP instead of costing
    // a 25-bit adder in fabric.  floor() splits exactly across a power of two:
    //     floor((a + b*2^17) / 2^17) == floor(a / 2^17) + b
    // so this is bit-identical to subtracting after the cascade.
    localparam int ROUND_BIAS = ACC_ONE_I - ACC_HALF_I;
    localparam logic signed [PROD_W-1:0] ROUND_FOLD =
        -(PROD_W'(ROUND_BIAS) <<< FRAC_BITS);

    localparam logic signed [PROD_W-1:0] C_TERM0 =
        (PROD_W'(POLY_C0) <<< C_SHIFT) + STEP_ROUND + ROUND_FOLD;
    localparam logic signed [PROD_W-1:0] C_TERM1 =
        (PROD_W'(POLY_C1) <<< C_SHIFT) + STEP_ROUND;
    localparam logic signed [PROD_W-1:0] C_TERM2 =
        (PROD_W'(POLY_C2) <<< C_SHIFT) + STEP_ROUND;
    localparam logic signed [PROD_W-1:0] C_TERM3 =
        (PROD_W'(POLY_C3) <<< C_SHIFT) + STEP_ROUND;

    // Horner seed: c4 rescaled from COEF_FRAC to ACC_FRAC.
    localparam logic signed [ACC_W-1:0] ACC_SEED =
        ACC_W'(POLY_C4) <<< (ACC_FRAC - COEF_FRAC);

    typedef enum logic [1:0] {
        ROUTE_POLY = 2'b00,
        ROUTE_TAIL = 2'b01,
        ROUTE_ZERO = 2'b10
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
    // Combinational: the early-out code is registered by the shared stage
    // register further down, so that it stays aligned with either front-end.
    early_out_t eo_comb;
    bf16_early_out #(.REGISTER_OUTPUT(1'b0)) u_early_out (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .decomposed(s1_decomposed), .eo_code(eo_comb)
    );

    // The polynomial covers everything up to the point where exp(x) stops being
    // a normal BF16 number; below that a small ROM holds the subnormal results.
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
            route_comb = ROUTE_POLY;
    end

    logic [MANT_SRC_W-1:0] mant_ext_comb;
    assign mant_ext_comb = {1'b1, s1_decomposed.mantissa};

    logic [3:0] tail_addr_comb;
    assign tail_addr_comb = 4'(s1_decomposed.mantissa - 7'(TAIL_MANT_LO));

    // -------------------------------------------------------------------------
    // Front end: aligned = ((1.mantissa) * log2(e)) >> (13 - exp)
    // -------------------------------------------------------------------------
    // Two structurally different circuits computing the same integer function.
    // They are bit-identical and differ only in which resource they consume:
    //
    //   DSP_FRONTEND = 0   CSD adder tree + barrel shifter    0 DSP, ~130 LUT
    //   DSP_FRONTEND = 1   two DSP48 slices                   2 DSP,  ~20 LUT
    //
    // Both operations are kept as plain multiplies, because that is the only
    // pattern Vivado reliably maps to a DSP48E1.  Anything shaped like
    // "constant * x + y" is taken as an invitation to build a CSD adder tree in
    // fabric and use the DSP as a bare adder, which costs far more LUTs than
    // the barrel shifter it was meant to replace.
    //
    //   DSP #1   t       = (1.mantissa) * LOG2E_Q          A = 24 b, B = 8 b
    //   DSP #2   aligned = ((t >> 7) * 2^(e+9)) >> 15      A = 25 b, B = 16 b
    //
    // The second multiply is the variable right shift written as a multiply by
    // a one-hot constant.  Pre-shifting t by a fixed 7 bits is what makes the A
    // operand fit the 25-bit port, and it is lossless because the required
    // shift 13 - e is never smaller than 7 on the polynomial path:
    //     (t >> (13-e))  ==  ((t >> 7) * 2^(e+9)) >> 15     for e <= 6
    // Verified exhaustively against the barrel shifter: 0 differences over the
    // 2048 inputs that reach the polynomial.  e = 7 is flushed to zero, so the
    // 4-bit one-hot select deliberately does not cover it.
    localparam int MULT_T_W  = CONST_MULT_W;           // 8 x 24
    localparam int T7_SHIFT  = 7;
    localparam int T7_W      = MULT_T_W - T7_SHIFT;    // 25 -> DSP A port
    localparam int ONEHOT_W  = 16;                     // e + 9 in [0, 15]
    localparam int SCALED_W  = T7_W + ONEHOT_W;
    localparam int OUT_SHIFT = 15;

    // Everything below is valid in the same cycle as fe_aligned.
    logic [ALIGNED_W-1:0] fe_aligned;
    route_t               fe_route;
    logic [3:0]           fe_tail_addr;
    early_out_t           fe_eo;

    generate
    if (DSP_FRONTEND) begin : gen_frontend_dsp
        // DSP #1: the constant multiply, forced into the block.
        (* use_dsp = "yes" *) logic [MULT_T_W-1:0] mult_t;
        assign mult_t = mant_ext_comb * LOG2E_Q;

        logic [T7_W-1:0] t7_comb;
        assign t7_comb = mult_t[MULT_T_W-1:T7_SHIFT];

        // exp is in [-9, 6] on this path; +9 turns it into a 4-bit decode.
        /* verilator lint_off UNUSEDSIGNAL */
        logic signed [8:0] onehot_sel;
        /* verilator lint_on UNUSEDSIGNAL */
        logic [ONEHOT_W-1:0] onehot_comb;
        assign onehot_sel  = $signed(s1_decomposed.exponent) + 9'sd9;
        assign onehot_comb = ONEHOT_W'(1'b1) << onehot_sel[3:0];

        logic [T7_W-1:0]     fe1_t7;
        logic [ONEHOT_W-1:0] fe1_onehot;
        route_t              fe1_route;
        logic [3:0]          fe1_tail;
        early_out_t          fe1_eo;

        if (REGISTER_STAGES && RESET_DATAPATH) begin : gen_fe_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    fe1_t7     <= '0;
                    fe1_onehot <= '0;
                    fe1_route  <= ROUTE_ZERO;
                    fe1_tail   <= '0;
                    fe1_eo     <= EO_PLUS_ONE;
                end else if (pipe_en) begin
                    fe1_t7     <= t7_comb;
                    fe1_onehot <= onehot_comb;
                    fe1_route  <= route_comb;
                    fe1_tail   <= tail_addr_comb;
                    fe1_eo     <= eo_comb;
                end
            end
        end else if (REGISTER_STAGES) begin : gen_fe_reg
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    fe1_t7     <= t7_comb;
                    fe1_onehot <= onehot_comb;
                    fe1_route  <= route_comb;
                    fe1_tail   <= tail_addr_comb;
                    fe1_eo     <= eo_comb;
                end
            end
        end else begin : gen_fe_wire
            assign fe1_t7     = t7_comb;
            assign fe1_onehot = onehot_comb;
            assign fe1_route  = route_comb;
            assign fe1_tail   = tail_addr_comb;
            assign fe1_eo     = eo_comb;
        end

        // DSP #2: the alignment shift, again as a plain multiply.
        (* use_dsp = "yes" *) logic [SCALED_W-1:0] scaled_comb;
        assign scaled_comb = fe1_t7 * fe1_onehot;

        assign fe_aligned   = scaled_comb[OUT_SHIFT +: ALIGNED_W];
        assign fe_route     = fe1_route;
        assign fe_tail_addr = fe1_tail;
        assign fe_eo        = fe1_eo;
    end else begin : gen_frontend_barrel
        // One operand is a constant, so synthesis maps this to a CSD adder tree
        // rather than a DSP block.
        logic [CONST_MULT_W-1:0] mult_comb;
        assign mult_comb = mant_ext_comb * LOG2E_Q;

        // Single variable shift places the binary point.  shift_signed is
        // computed at full width and then narrowed; the upper bits are
        // sign/guard only, which is why they are intentionally unused.
        /* verilator lint_off UNUSEDSIGNAL */
        logic signed [8:0] shift_signed;
        /* verilator lint_on UNUSEDSIGNAL */
        logic [4:0] shift_comb;
        assign shift_signed = $signed(9'(LOG2E_FRAC_BITS + 7 - FRAC_BITS))
                            - $signed(s1_decomposed.exponent);
        assign shift_comb   = shift_signed[4:0];  // in [6, 22] for exp in [-9, 7]

        assign fe_aligned   = ALIGNED_W'(mult_comb >> shift_comb);
        assign fe_route     = route_comb;
        assign fe_tail_addr = tail_addr_comb;
        assign fe_eo        = eo_comb;
    end
    endgenerate

    logic [INT_W-1:0]     int_part_comb;
    logic [FRAC_BITS-1:0] frac_comb;
    assign int_part_comb = fe_aligned[ALIGNED_W-1:FRAC_BITS];
    assign frac_comb     = fe_aligned[FRAC_BITS-1:0];

    route_t               s2_route;
    early_out_t           s2_eo;
    logic [INT_W-1:0]     s2_int_part;
    logic [FRAC_BITS-1:0] s2_frac;
    logic [3:0]           s2_tail_addr;

    generate
        if (REGISTER_STAGES && RESET_DATAPATH) begin : gen_stage2_regs_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s2_route     <= ROUTE_ZERO;
                    s2_eo        <= EO_PLUS_ONE;
                    s2_int_part  <= '0;
                    s2_frac      <= '0;
                    s2_tail_addr <= '0;
                end else if (pipe_en) begin
                    s2_route     <= fe_route;
                    s2_eo        <= fe_eo;
                    s2_int_part  <= int_part_comb;
                    s2_frac      <= frac_comb;
                    s2_tail_addr <= fe_tail_addr;
                end
            end
        end else if (REGISTER_STAGES) begin : gen_stage2_regs
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    s2_route     <= fe_route;
                    s2_eo        <= fe_eo;
                    s2_int_part  <= int_part_comb;
                    s2_frac      <= frac_comb;
                    s2_tail_addr <= fe_tail_addr;
                end
            end
        end else begin : gen_stage2_wires
            assign s2_route     = fe_route;
            assign s2_eo        = fe_eo;
            assign s2_int_part  = int_part_comb;
            assign s2_frac      = frac_comb;
            assign s2_tail_addr = fe_tail_addr;
        end
    endgenerate

    // The subnormal tail is 11 x 7 bits, so it is read asynchronously and then
    // travels down the same delay line as the rest of the control state.
    logic [6:0] tail_data_comb;
    bf16_expe_poly4_tail_rom #(.REGISTERED(1'b0)) u_tail_rom (
        .clk(clk), .pipe_en(pipe_en),
        .addr(s2_tail_addr), .data(tail_data_comb)
    );

    // -------------------------------------------------------------------------
    // Stages 3 .. 3+DEGREE-1: the DSP cascade
    // -------------------------------------------------------------------------
    // Control state and f must ride alongside the accumulator, one delay per
    // Horner step, so that everything realigns at the packing stage.
    early_out_t           ctrl_eo   [0:DEGREE];
    route_t               ctrl_route[0:DEGREE];
    logic [INT_W-1:0]     ctrl_int  [0:DEGREE];
    logic [6:0]           ctrl_tail [0:DEGREE];
    logic [FRAC_BITS-1:0] ctrl_frac [0:DEGREE];

    assign ctrl_eo[0]    = s2_eo;
    assign ctrl_route[0] = s2_route;
    assign ctrl_int[0]   = s2_int_part;
    assign ctrl_tail[0]  = tail_data_comb;
    assign ctrl_frac[0]  = s2_frac;

    genvar gi;
    generate
        for (gi = 0; gi < DEGREE; gi++) begin : gen_ctrl_delay
            if (REGISTER_STAGES && RESET_DATAPATH) begin : gen_reg_rst
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        ctrl_eo[gi+1]    <= EO_PLUS_ONE;
                        ctrl_route[gi+1] <= ROUTE_ZERO;
                        ctrl_int[gi+1]   <= '0;
                        ctrl_tail[gi+1]  <= '0;
                        ctrl_frac[gi+1]  <= '0;
                    end else if (pipe_en) begin
                        ctrl_eo[gi+1]    <= ctrl_eo[gi];
                        ctrl_route[gi+1] <= ctrl_route[gi];
                        ctrl_int[gi+1]   <= ctrl_int[gi];
                        ctrl_tail[gi+1]  <= ctrl_tail[gi];
                        ctrl_frac[gi+1]  <= ctrl_frac[gi];
                    end
                end
            end else if (REGISTER_STAGES) begin : gen_reg
                // Without a reset these collapse into SRL16E shift registers,
                // which is both smaller and faster than a chain of flops.
                always_ff @(posedge clk) begin
                    if (pipe_en) begin
                        ctrl_eo[gi+1]    <= ctrl_eo[gi];
                        ctrl_route[gi+1] <= ctrl_route[gi];
                        ctrl_int[gi+1]   <= ctrl_int[gi];
                        ctrl_tail[gi+1]  <= ctrl_tail[gi];
                        ctrl_frac[gi+1]  <= ctrl_frac[gi];
                    end
                end
            end else begin : gen_wire
                // Drive every stage straight from the stage-2 signals instead of
                // chaining element to element.  With no registers the two are
                // identical, but reading and writing the same unpacked array in
                // combinational assigns makes Verilator report a false circular
                // dependency (UNOPTFLAT).
                assign ctrl_eo[gi+1]    = s2_eo;
                assign ctrl_route[gi+1] = s2_route;
                assign ctrl_int[gi+1]   = s2_int_part;
                assign ctrl_tail[gi+1]  = tail_data_comb;
                assign ctrl_frac[gi+1]  = s2_frac;
            end
        end
    endgenerate

    logic signed [ACC_W-1:0]  acc_chain [0:DEGREE];
    logic signed [PROD_W-1:0] c_terms   [0:DEGREE-1];

    assign acc_chain[0] = ACC_SEED;
    // Horner consumes coefficients from the highest remaining degree downwards.
    assign c_terms[0] = C_TERM3;
    assign c_terms[1] = C_TERM2;
    assign c_terms[2] = C_TERM1;
    assign c_terms[3] = C_TERM0;

    generate
        for (gi = 0; gi < DEGREE; gi++) begin : gen_horner
            bf16_expe_poly4_step #(
                .REGISTERED(REGISTER_STAGES),
                .RESET_DATAPATH(RESET_DATAPATH)
            ) u_step (
                .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
                .acc_in(acc_chain[gi]),
                .frac(ctrl_frac[gi]),
                .c_term(c_terms[gi]),
                .acc_out(acc_chain[gi+1])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Final stage: round to 7 mantissa bits and pack
    // -------------------------------------------------------------------------
    // ROUND_FOLD already applied the "- 1.0 + half LSB" inside the last DSP,
    // so the cascade output is the rounding sum itself.
    logic [7:0] count;
    assign count = 8'(acc_chain[DEGREE] >>> ROUND_SHIFT);

    // exponent = -I-1, biased = 127 + (-I-1) = 126 - I.
    // count == 128 means the mantissa rounded up to 2.0: mantissa 0, exponent +1.
    // count[6:0] is already 0 in that case, so only the exponent needs the carry.
    logic [7:0] biased_exp;
    logic [6:0] out_mantissa;
    assign biased_exp   = 8'd126 - 8'(ctrl_int[DEGREE]) + 8'(count[7]);
    assign out_mantissa = count[6:0];

    logic [15:0] poly_result;
    assign poly_result = {1'b0, biased_exp, out_mantissa};

    // -------------------------------------------------------------------------
    // Early-out and route mux
    // -------------------------------------------------------------------------
    logic [15:0] result_comb;
    always_comb begin
        unique case (ctrl_eo[DEGREE])
            EO_QNAN:      result_comb = BF16_QNAN;
            EO_PLUS_ONE:  result_comb = BF16_PLUS_ONE;
            EO_PLUS_ZERO: result_comb = BF16_PLUS_ZERO;
            default: begin
                unique case (ctrl_route[DEGREE])
                    ROUTE_POLY: result_comb = poly_result;
                    ROUTE_TAIL: result_comb = {9'b0, ctrl_tail[DEGREE]};
                    default:    result_comb = 16'h0000;
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
        .STAGES        (PAD_STAGES),
        .WIDTH         (16),
        .RESET_DATAPATH(RESET_DATAPATH)
    ) u_pipe_pad (
        .clk(clk), .rst_n(rst_n), .pipe_en(pipe_en),
        .d(core_data), .q(m_axis_tdata)
    );
endmodule : bf16_expe_poly4

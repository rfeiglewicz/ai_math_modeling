// =============================================================================
// bf16_exp2_optim_approx.sv
// Piecewise linear approximation of 2^(-x) for x in [0, 1):  res = b - a*x
//
// This is where most of the area saving is. Against bf16_linear_approx:
//
//   multiplier   21 x 30 cascaded over two DSP48 slices  ->  17 x 17, one slice
//   subtractor   62-bit signed with four integer guards  ->  18-bit unsigned
//   ROM word     42 bits                                 ->  35 bits
//
// and the partial-product split (SPLIT_MULT) disappears with the cascade it
// existed to break.
//
// -----------------------------------------------------------------------------
// The narrow subtraction
//
// a*x carries PROD_F = 34 fractional bits, b carries B_F = 18. The naive form
// aligns both to Q0.34, subtracts, and throws away the low 16 bits of the
// result -- a 34-bit borrow chain of which half is discarded.
//
// Because the low CDROP bits of the aligned b are zero, the same value comes
// out of an 18-bit subtractor:
//
//     floor(((b << 16) - ax) / 2^16) = b - ceil(ax / 2^16)
//                                    = b - (ax >> 16) - (ax[15:0] != 0)
//
// so: one 18-bit subtract, a 16-input OR reduce, and a borrow-in. The OR is a
// two-level LUT tree and runs in parallel with the subtract's own carry chain.
//
// -----------------------------------------------------------------------------
// Why the result needs no sign bit and no integer bits
//
// max(a) = 0.690553, max(x) < 1 and min(b) = 0.847980 over the whole BF16
// input space, and the measured max(a*x) is 0.347323. So b - a*x is in
// [0.500657, 0.999992] and cannot go negative or reach 1.0. Q0.18 unsigned.
// =============================================================================

module bf16_exp2_optim_approx
    import bf16_exp2_optim_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Retiming: extra pipeline registers inside the arithmetic.
    //   0 - ROM read, multiply and subtract all share one stage.
    //   1 - register the raw product, splitting multiply from subtract. The
    //       register lands in the DSP48 PREG, so it costs no fabric.
    //   2 - additionally register the subtract result, so the borrow chain no
    //       longer shares a stage with the normaliser.
    // Purely a timing transformation; the result stays bit-identical.
    parameter int EXTRA_STAGES    = 0
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic              pipe_en,
    input  logic [X_W-1:0]    x,     // Q0.17 unsigned
    output logic [CALC_W-1:0] res    // Q0.18 unsigned, in [0.5, 1)
);

    // -------------------------------------------------------------------------
    // Segment index: top LUT_ADDR_W bits of x, inverted so that the decreasing
    // 2^-x maps onto an increasing table.
    // -------------------------------------------------------------------------
    logic [LUT_ADDR_W-1:0] lut_idx_raw;
    logic [LUT_ADDR_W-1:0] lut_idx;

    assign lut_idx_raw = x[X_F-1 -: LUT_ADDR_W];
    assign lut_idx     = LUT_ADDR_W'(LUT_ADDR_W'(LUT_SIZE - 1) - lut_idx_raw);

    // -------------------------------------------------------------------------
    // Coefficient ROM.
    //   REGISTERED=1 -> synchronous output, inferred as BRAM, 1-cycle latency
    //   REGISTERED=0 -> combinational output, LUT ROM, 0-cycle latency
    // -------------------------------------------------------------------------
    logic [PACKED_W-1:0] packed_coeff;

    bf16_exp2_optim_rom #(
        .DATA_W    (PACKED_W),
        .DEPTH     (LUT_SIZE),
        .ADDR_W    (LUT_ADDR_W),
        .REGISTERED(REGISTER_OUTPUT)
    ) u_coeff_rom (
        .clk    (clk),
        .pipe_en(pipe_en),
        .addr   (lut_idx),
        .data   (packed_coeff)
    );

    logic [A_W-1:0] coeff_a_w;
    logic [B_W-1:0] coeff_b_w;

    assign coeff_a_w = packed_coeff[A_W-1 : 0];
    assign coeff_b_w = packed_coeff[PACKED_W-1 : A_W];

    // -------------------------------------------------------------------------
    // x alignment: when the ROM is synchronous its output is one cycle late,
    // so x has to be delayed by the same amount before it meets the coefficients.
    // -------------------------------------------------------------------------
    logic [X_W-1:0] x_aligned;

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_x_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       x_aligned <= '0;
                else if (pipe_en) x_aligned <= x;
            end
        end else if (REGISTER_OUTPUT) begin : gen_x_delay_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) x_aligned <= x;
            end
        end else begin : gen_x_wire
            assign x_aligned = x;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // a * x : Q0.17 * Q0.17 = Q0.34. Both operands fit the DSP48E1 A (25) and
    // B (18) ports directly, so this is one slice with no cascade.
    // -------------------------------------------------------------------------
    logic [PROD_W-1:0] ax_comb;
    logic [PROD_W-1:0] ax;
    logic [B_W-1:0]    coeff_b_stage;

    (* use_dsp = "yes" *) logic [PROD_W-1:0] ax_dsp;
    assign ax_dsp  = coeff_a_w * x_aligned;
    assign ax_comb = ax_dsp;

    generate
        if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_mult_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    ax            <= '0;
                    coeff_b_stage <= '0;
                end else if (pipe_en) begin
                    ax            <= ax_comb;
                    coeff_b_stage <= coeff_b_w;
                end
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_mult_reg
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    ax            <= ax_comb;
                    coeff_b_stage <= coeff_b_w;
                end
            end
        end else begin : gen_mult_wire
            assign ax            = ax_comb;
            assign coeff_b_stage = coeff_b_w;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // res = b - (ax >> CDROP) - (ax[CDROP-1:0] != 0)
    // -------------------------------------------------------------------------
    logic [CALC_W-1:0] ax_hi;
    logic              ax_lo_nonzero;
    logic [CALC_W-1:0] res_comb;

    assign ax_hi         = ax[PROD_W-1 -: CALC_W];
    assign ax_lo_nonzero = |ax[CDROP-1:0];
    assign res_comb      = coeff_b_stage - ax_hi - CALC_W'(ax_lo_nonzero);

    generate
        if (EXTRA_STAGES >= 2 && RESET_DATAPATH) begin : gen_sub_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       res <= '0;
                else if (pipe_en) res <= res_comb;
            end
        end else if (EXTRA_STAGES >= 2) begin : gen_sub_reg
            always_ff @(posedge clk) begin
                if (pipe_en) res <= res_comb;
            end
        end else begin : gen_sub_wire
            assign res = res_comb;
        end
    endgenerate

endmodule : bf16_exp2_optim_approx

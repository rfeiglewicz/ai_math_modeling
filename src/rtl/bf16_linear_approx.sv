// =============================================================================
// bf16_linear_approx.sv
// Piecewise linear approximation of 2^(-x) for x in [0, 1).
// Formula: result = b - a*x
//
// Coefficients are stored in a separate bf16_coeff_rom module.
// When REGISTER_OUTPUT=1 the ROM is inferred as BRAM (synchronous read).
// frac_part is then delayed by 1 cycle to stay aligned with ROM output,
// so the multiply always sees data from the same input cycle.
//
// Pipeline latency:
//   REGISTER_OUTPUT=0 : 0 cycles  (combinational LUT ROM, async multiply)
//   REGISTER_OUTPUT=1 : 1 cycle   (BRAM read is the pipeline register stage)
//
// Parameterizable:
//   COEFF_W  - coefficient width  (default 21, format 1.20)
//   COEFF_F  - coefficient frac   (default 20)
//   CALC_W   - result width       (default 62, format 4.58 signed)
// =============================================================================

module bf16_linear_approx
    import bf16_exp2_pkg::*;
#(
    parameter int COEFF_W         = COEFF_W_DEFAULT,  // 21
    parameter int COEFF_F         = COEFF_F_DEFAULT,  // 20
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Number of LOW bits of frac_part that are guaranteed to be zero.
    // Set from MANT_MULT_ROUND_FRAC at the top level: rounding the log2(e)
    // product to 2.F leaves (29-F) zero LSBs, and the unified shift is a pure
    // LEFT shift, so those zeros survive into frac_part unchanged.
    // Narrowing the multiplier by this amount is exact, not an approximation.
    parameter int FRAC_ZERO_LSBS  = 0
)(
    input  logic              clk,
    input  logic              rst_n,
    input  logic              pipe_en,
    input  logic [IN_F-1:0]   frac_part,         // 38-bit input x (1.37 unsigned)
    output logic [CALC_W-1:0] unnormalized_res   // 62-bit result  (4.58 signed)
);

    localparam int PACKED_W       = 2 * COEFF_W;      // 42
    localparam int ACTUAL_MULT_W  = IN_F + COEFF_W;   // 59  (2.57 unsigned)
    localparam int B_SHIFT        = CALC_F - COEFF_F; // 38  (align b to CALC_F=58)

    // -------------------------------------------------------------------------
    // LUT index  (7 MSBs of frac_part, inverted)
    // C++: idx = LUT_MAX_IDX - frac_part[IN_F-1 -: LUT_ADDR_W]
    // -------------------------------------------------------------------------
    logic [LUT_ADDR_W-1:0] lut_idx_raw;
    logic [LUT_ADDR_W-1:0] lut_idx;

    assign lut_idx_raw = frac_part[IN_F-1 -: LUT_ADDR_W];
    assign lut_idx     = 7'(7'(LUT_SIZE - 1) - lut_idx_raw);

    // -------------------------------------------------------------------------
    // Coefficient ROM  (bf16_coeff_rom)
    //   REGISTERED=1  -> synchronous output, inferred as BRAM, 1-cycle latency
    //   REGISTERED=0  -> combinational output, LUT ROM, 0-cycle latency
    // -------------------------------------------------------------------------
    logic [PACKED_W-1:0] packed_coeff;

    bf16_coeff_rom #(
        .DATA_W    (PACKED_W),
        .DEPTH     (LUT_SIZE),
        .ADDR_W    (LUT_ADDR_W),
        .REGISTERED(REGISTER_OUTPUT)
    ) u_coeff_rom (
        .clk     (clk),
        .pipe_en (pipe_en),
        .addr    (lut_idx),
        .data    (packed_coeff)
    );

    logic [COEFF_W-1:0] coeff_a_w;
    logic [COEFF_W-1:0] coeff_b_w;

    assign coeff_a_w = packed_coeff[COEFF_W-1 : 0];
    assign coeff_b_w = packed_coeff[PACKED_W-1 : COEFF_W];

    // -------------------------------------------------------------------------
    // frac_part alignment
    // When REGISTER_OUTPUT=1 the ROM output is 1 cycle late (BRAM), so we
    // must also delay frac_part by 1 cycle before feeding the multiplier.
    // When REGISTER_OUTPUT=0 the ROM is combinational; no delay needed.
    // -------------------------------------------------------------------------
    logic [IN_F-1:0] frac_aligned;

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_frac_delay
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)      frac_aligned <= '0;
                else if (pipe_en) frac_aligned <= frac_part;
            end
        end else if (REGISTER_OUTPUT) begin : gen_frac_delay_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) frac_aligned <= frac_part;
            end
        end else begin : gen_frac_wire
            assign frac_aligned = frac_part;
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Computation  result = b - a * x   (combinational)
    //
    // a, b : 1.20 unsigned  (COEFF_W=21)
    // x    : 1.37 unsigned  (IN_F=38)
    // a*x  : 2.57 unsigned  (ACTUAL_MULT_W=59)
    // b aligned to CALC_W=62 at bit CALC_F=58 : b << B_SHIFT(38)
    //
    // The low FRAC_ZERO_LSBS bits of x are known to be zero, so
    //     a * x = a * (x_hi << FRAC_ZERO_LSBS) = (a * x_hi) << FRAC_ZERO_LSBS
    // and the multiplier only needs the upper MULT_X_W bits. This removes whole
    // DSP48 slices and the fabric adders that merge them.
    // -------------------------------------------------------------------------
    localparam int MULT_X_W    = IN_F - FRAC_ZERO_LSBS;      // 38 or 30
    localparam int CORE_MULT_W = COEFF_W + MULT_X_W;         // 59 or 51

    logic [MULT_X_W-1:0]      frac_hi;
    logic [CORE_MULT_W-1:0]   ax_core;
    logic [ACTUAL_MULT_W-1:0] ax_unsigned;
    logic signed [CALC_W-1:0] neg_ax;
    logic signed [CALC_W-1:0] b_aligned;
    logic signed [CALC_W-1:0] calc_res;

    assign frac_hi = frac_aligned[IN_F-1 : FRAC_ZERO_LSBS];

    always_comb begin
        ax_core     = coeff_a_w * frac_hi;
        ax_unsigned = ACTUAL_MULT_W'(ax_core) << FRAC_ZERO_LSBS;
        neg_ax      = -signed'({1'b0, ax_unsigned});
        b_aligned   = signed'({1'b0, CALC_W'({coeff_b_w, {B_SHIFT{1'b0}}})});
        calc_res    = b_aligned + neg_ax;
    end

    // -------------------------------------------------------------------------
    // Output assignment
    // When REGISTER_OUTPUT=1 the BRAM read IS the pipeline register stage --
    // no additional always_ff here so stage latency stays at exactly 1 cycle.
    // When REGISTER_OUTPUT=0 the result is purely combinational.
    // -------------------------------------------------------------------------
    assign unnormalized_res = calc_res;

endmodule : bf16_linear_approx

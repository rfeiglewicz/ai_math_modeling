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
    parameter int FRAC_ZERO_LSBS  = 0,
    // Retiming: extra pipeline registers inside the arithmetic.
    //   0 - ROM read, multiply and subtract all share one stage (original).
    //   1 - register the raw product, splitting multiply from subtract. The
    //       register lands in the DSP48 PREG, so it costs no fabric.
    //   2 - additionally register the subtract result.
    // Purely a timing transformation: the arithmetic is untouched, so the
    // result stays bit-identical. Only latency grows, by EXTRA_STAGES cycles.
    parameter int EXTRA_STAGES    = 0,
    // Split the multiply into partial products that each fit one DSP48E1.
    //
    // COEFF_W x MULT_X_W is 21 x 30, and the DSP48E1 ports are A = 25 and
    // B = 18, so the wide operand does not fit. Vivado therefore cascades two
    // slices through PCOUT -> PCIN, and because there is no register anywhere
    // in that cascade the whole thing is one combinational hop: the multiply,
    // the cascade route and the partial-product add in series. That was the
    // critical path once everything else had been split.
    //
    //     a * x = sum_i  (a * x_i) << (i * SPLIT_W)      x_i = SPLIT_W bits
    //
    // Each a * x_i fits a single slice, so the register from EXTRA_STAGES can
    // sit on the partial products (DSP MREG/PREG) instead of after the merged
    // result. Exact by construction - it is just the distributive law, no bits
    // are dropped - so the output stays bit-identical.
    // Requires EXTRA_STAGES >= 1 to be useful; ignored otherwise.
    parameter bit SPLIT_MULT      = 1'b0
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
    logic [CORE_MULT_W-1:0]   ax_core_comb;
    logic [CORE_MULT_W-1:0]   ax_core;
    logic [COEFF_W-1:0]       coeff_b_stage;
    logic [ACTUAL_MULT_W-1:0] ax_unsigned;
    logic signed [CALC_W-1:0] neg_ax;
    logic signed [CALC_W-1:0] b_aligned;
    logic signed [CALC_W-1:0] calc_res;

    assign frac_hi      = frac_aligned[IN_F-1 : FRAC_ZERO_LSBS];
    assign ax_core_comb = coeff_a_w * frac_hi;

    // -------------------------------------------------------------------------
    // Retiming register 1: between the multiply and the subtract.
    // b has to be delayed with it so both operands still meet on the same
    // input sample. Vivado folds this register into the DSP48 PREG.
    //
    // With SPLIT_MULT the register moves one step earlier, onto the individual
    // partial products, which is what breaks the DSP-to-DSP cascade.
    // -------------------------------------------------------------------------
    // Widest B-port operand a single DSP48E1 takes is 18 bits; 15 keeps some
    // headroom and divides 30 exactly.
    localparam int SPLIT_W  = 15;
    localparam int NSPLIT   = (MULT_X_W + SPLIT_W - 1) / SPLIT_W;
    localparam int PAD_X_W  = NSPLIT * SPLIT_W;
    localparam int PART_W   = COEFF_W + SPLIT_W;

    generate
        if (SPLIT_MULT && EXTRA_STAGES >= 1) begin : gen_split_mult
            logic [PAD_X_W-1:0]     frac_padded;
            logic [PART_W-1:0]      part_comb [NSPLIT];
            logic [PART_W-1:0]      part_reg  [NSPLIT];
            logic [CORE_MULT_W-1:0] merged;

            assign frac_padded = PAD_X_W'(frac_hi);

            for (genvar i = 0; i < NSPLIT; i++) begin : gen_part
                // One clean A*B per slice, so it maps to a DSP without cascade.
                (* use_dsp = "yes" *) logic [PART_W-1:0] p;
                assign p = coeff_a_w * frac_padded[i*SPLIT_W +: SPLIT_W];
                assign part_comb[i] = p;

                if (RESET_DATAPATH) begin : gen_part_reg_rst
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n)       part_reg[i] <= '0;
                        else if (pipe_en) part_reg[i] <= part_comb[i];
                    end
                end else begin : gen_part_reg
                    always_ff @(posedge clk) begin
                        if (pipe_en) part_reg[i] <= part_comb[i];
                    end
                end
            end

            // Recombine. The parts do overlap, so this is a real add, but it is
            // only PART_W wide rather than the full product width.
            always_comb begin
                merged = '0;
                for (int i = 0; i < NSPLIT; i++)
                    merged += CORE_MULT_W'(part_reg[i]) << (i * SPLIT_W);
            end

            // ---------------------------------------------------------------
            // Retiming register 3 (EXTRA_STAGES >= 3): between the merge adder
            // and the subtract. Without it the merge carry chain and the
            // 62-bit subtract carry chain run back to back -- measured as 11
            // CARRY4 in series, 6.34 ns. dont_touch stops Vivado from packing
            // it back into the DSP output path.
            // coeff_b needs the matching extra delay.
            // ---------------------------------------------------------------
            if (EXTRA_STAGES >= 3) begin : gen_merge_reg
                (* dont_touch = "true" *) logic [CORE_MULT_W-1:0] merged_q;
                logic [COEFF_W-1:0] coeff_b_d;

                if (RESET_DATAPATH) begin : gen_rst
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            merged_q      <= '0;
                            coeff_b_d     <= '0;
                            coeff_b_stage <= '0;
                        end else if (pipe_en) begin
                            merged_q      <= merged;
                            coeff_b_d     <= coeff_b_w;
                            coeff_b_stage <= coeff_b_d;
                        end
                    end
                end else begin : gen_nrst
                    always_ff @(posedge clk) begin
                        if (pipe_en) begin
                            merged_q      <= merged;
                            coeff_b_d     <= coeff_b_w;
                            coeff_b_stage <= coeff_b_d;
                        end
                    end
                end

                assign ax_core = merged_q;
            end else begin : gen_merge_wire
                assign ax_core = merged;

                if (RESET_DATAPATH) begin : gen_b_reg_rst
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n)       coeff_b_stage <= '0;
                        else if (pipe_en) coeff_b_stage <= coeff_b_w;
                    end
                end else begin : gen_b_reg
                    always_ff @(posedge clk) begin
                        if (pipe_en) coeff_b_stage <= coeff_b_w;
                    end
                end
            end
        end else if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_mult_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    ax_core       <= '0;
                    coeff_b_stage <= '0;
                end else if (pipe_en) begin
                    ax_core       <= ax_core_comb;
                    coeff_b_stage <= coeff_b_w;
                end
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_mult_reg
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    ax_core       <= ax_core_comb;
                    coeff_b_stage <= coeff_b_w;
                end
            end
        end else begin : gen_mult_wire
            assign ax_core       = ax_core_comb;
            assign coeff_b_stage = coeff_b_w;
        end
    endgenerate

    always_comb begin
        ax_unsigned = ACTUAL_MULT_W'(ax_core) << FRAC_ZERO_LSBS;
        neg_ax      = -signed'({1'b0, ax_unsigned});
        b_aligned   = signed'({1'b0, CALC_W'({coeff_b_stage, {B_SHIFT{1'b0}}})});
        calc_res    = b_aligned + neg_ax;
    end

    // -------------------------------------------------------------------------
    // Retiming register 2: after the 62-bit subtract, so the carry chain no
    // longer shares a stage with the priority encoder in bf16_normalize.
    // -------------------------------------------------------------------------
    generate
        if (EXTRA_STAGES >= 2 && RESET_DATAPATH) begin : gen_sub_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       unnormalized_res <= '0;
                else if (pipe_en) unnormalized_res <= calc_res;
            end
        end else if (EXTRA_STAGES >= 2) begin : gen_sub_reg
            always_ff @(posedge clk) begin
                if (pipe_en) unnormalized_res <= calc_res;
            end
        end else begin : gen_sub_wire
            assign unnormalized_res = calc_res;
        end
    endgenerate

endmodule : bf16_linear_approx

// =============================================================================
// bf16_exp2_optim_normalize.sv
// Normalises the polynomial result to Q1.16.
//
// This module replaces bf16_normalize, which is a 62-bit priority encoder
// followed by a 62-bit barrel shifter -- together the deepest logic in the
// production core. Here it is one bit test and a 2:1 mux, because the range
// of b - a*x is known.
//
//   STATIC_NORM = 1   mantissa = res << 1,               exponent = -1
//   STATIC_NORM = 0   mantissa = res << (res[15] ? 1:2), exponent = res[15] ? -1 : -2
//
// -----------------------------------------------------------------------------
// Which one to build
//
// The 64-entry coefficient table is fitted with b - a*x in [0.5, 1) as a hard
// constraint on every reachable input, and tests/bf16_exp2_optim_test.cpp
// measures [0.500188, 0.999985] over the whole 2^16 domain of x, reachable or
// not. So with the current table the static form is unconditionally correct.
// (The 128-entry predecessor undershot by one LSB at the single largest,
// unreachable code of x.) The claim still has to be re-proved whenever the
// table, MANT_MULT_F, X_F or the input exponent range changes.
//
// The guarded form is the default because it is cheap, not because it is free.
// res in [0.5, 1) shifted left by 1 and res in [0.25, 0.5) shifted left by 2
// both land in [2^16, 2^17), so the output is Q1.16 either way -- the guard
// does not buy an extra bit of WIDTH, only an extra bit test and a mux.
// Measured on xc7a200t (make sweep_optim): 369 vs 350 LUT for the whole core,
// so 19 LUT and 4 FF, with identical Fmax. That is ~5% of the core's logic to
// remove the obligation to re-prove reachability every time the table,
// MANT_MULT_F, X_F or the input exponent range changes.
//
// tests/bf16_exp2_optim_test.cpp enumerates the reachable input set and checks
// the two variants agree on all of it.
// =============================================================================

module bf16_exp2_optim_normalize
    import bf16_exp2_optim_pkg::*;
#(
    // 1 = assume res >= 0.5 and shift by a constant 1.
    // 0 = test the top bit and shift by 1 or 2 (default).
    parameter bit STATIC_NORM     = 1'b0,
    parameter bit REGISTER_OUTPUT = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Kept for symmetry with the production core's retiming knobs. The logic
    // here is a mux, so there is nothing worth splitting; the parameter exists
    // so pipeline depth arithmetic stays uniform across both cores.
    parameter int EXTRA_STAGES    = 0
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    pipe_en,
    input  logic [CALC_W-1:0]       res,              // Q0.16 unsigned
    output logic [POLY_OUT_W-1:0]   normalized_mant,  // Q1.16, in [1.0, 2.0)
    output logic signed [8:0]       poly_exponent
);

    logic                  top_bit;
    logic [POLY_OUT_W-1:0] mant_comb;
    logic signed [8:0]     exp_comb;

    assign top_bit = res[CALC_F-1];

    generate
        if (STATIC_NORM) begin : gen_static
            assign mant_comb = POLY_OUT_W'(res) << 1;
            assign exp_comb  = 9'(signed'(STATIC_POLY_EXP));
        end else begin : gen_guarded
            assign mant_comb = top_bit ? (POLY_OUT_W'(res) << 1)
                                       : (POLY_OUT_W'(res) << 2);
            assign exp_comb  = top_bit ? 9'(signed'(STATIC_POLY_EXP))
                                       : 9'(signed'(STATIC_POLY_EXP - 1));
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Optional retiming stage, kept so both cores count depth the same way.
    // -------------------------------------------------------------------------
    logic [POLY_OUT_W-1:0] mant_s;
    logic signed [8:0]     exp_s;

    generate
        if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_split_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    mant_s <= '0; exp_s <= '0;
                end else if (pipe_en) begin
                    mant_s <= mant_comb; exp_s <= exp_comb;
                end
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_split
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    mant_s <= mant_comb; exp_s <= exp_comb;
                end
            end
        end else begin : gen_split_wire
            assign mant_s = mant_comb;
            assign exp_s  = exp_comb;
        end
    endgenerate

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    normalized_mant <= '0;
                    poly_exponent   <= '0;
                end else if (pipe_en) begin
                    normalized_mant <= mant_s;
                    poly_exponent   <= exp_s;
                end
            end
        end else if (REGISTER_OUTPUT) begin : gen_reg_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    normalized_mant <= mant_s;
                    poly_exponent   <= exp_s;
                end
            end
        end else begin : gen_comb
            assign normalized_mant = mant_s;
            assign poly_exponent   = exp_s;
        end
    endgenerate

endmodule : bf16_exp2_optim_normalize

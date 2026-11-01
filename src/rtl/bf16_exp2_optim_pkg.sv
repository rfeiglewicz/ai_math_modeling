// =============================================================================
// bf16_exp2_optim_pkg.sv
// Bit widths for the width-optimised BF16 exp2 / expe core.
//
// Mirrors the C++ namespace bf16_optim_cfg in
// src/approximations/bf16_exp2_optim.hpp one-for-one. The widths come from
// tests/pwl_width_refit_study.cpp, which refits the 64-entry coefficient
// table for every candidate datapath and checks it exhaustively: this is the
// narrowest point on its grid that keeps the core bit-identical to
// bf16_exp2_approx<29> over all 65536 BF16 patterns in both modes, under
// both log2(e) roundings (ROUND_MODE 0 and 1). The table in
// bf16_exp2_optim_rom.sv is fitted for exactly these widths by
// modeling/coeff_gen/gen_optim_coeffs.cpp; change one and re-run it.
//
// This package is deliberately separate from bf16_exp2_pkg. The production
// constants are all anchored to MANT_MULT_F = 29, so sharing them would either
// drag the wide widths back in or require editing production code.
//
// The two packages are NOT wildcard-import compatible: they define many of the
// same names with different values. Modules in this core import this package
// only, and reach for bf16_exp2_pkg::fp_raw_t / ::early_out_t by explicit
// qualification where the shared plumbing modules (decompose, early_out,
// recompose, pipe_pad) require them. Those four are format-level glue, are
// identical in both cores, and are instantiated unchanged.
//
// Width summary (128-entry predecessor in brackets):
//   MANT_SRC_W  =  8  (Q1.7  unsigned)   unchanged from production
//   LOG2E_W     = 23  (Q1.22 unsigned)   unchanged from production
//   MANT_MULT_W = 20  (Q2.18 unsigned)   [23]  production 31
//   UNIFIED_W   = 27  (Q9.18 unsigned)   [30]  production 47
//   X_W         = 16  (Q0.16 unsigned)   [17]  production 38
//   A_W         = 11  (Q0.11 unsigned)   [17]  production 21 (Q1.20)
//   B_W         = 16  (Q0.16 unsigned)   [18]  production 21 (Q1.20)
//   PROD_W      = 27  (Q0.27 unsigned)   [34]  production 60
//   CALC_W      = 16  (Q0.16 unsigned)   [18]  production 62 signed
//   POLY_OUT_W  = 17  (Q1.16 unsigned)   [19]  production 59
//   PACKED_W    = 27  (ROM word)         [35]  production 42
//   LUT_SIZE    = 64                     [128] production 128
// =============================================================================

package bf16_exp2_optim_pkg;

    // =========================================================================
    // BF16 target format
    // =========================================================================
    localparam int BF16_MANT_BITS   = 7;
    localparam int BF16_BIAS        = 127;
    localparam int BF16_MIN_EXP     = 1 - BF16_BIAS;   // -126

    // =========================================================================
    // Input exponent range handled by the polynomial core.
    // Outside it bf16_early_out forces +1.0 or +0.0.
    // =========================================================================
    localparam int INPUT_MIN_EXP    = -9;
    localparam int INPUT_MAX_EXP    = 7;

    // =========================================================================
    // Mantissa source: hidden bit + 7 mantissa bits, always in [1.0, 2.0)
    // =========================================================================
    localparam int MANT_SRC_I       = 1;
    localparam int MANT_SRC_F       = BF16_MANT_BITS;             // 7
    localparam int MANT_SRC_W       = MANT_SRC_I + MANT_SRC_F;    // 8

    // =========================================================================
    // log2(e) constant: Q1.22, unchanged from the shipped table
    // =========================================================================
    localparam int LOG2E_I_DEFAULT   = 1;
    localparam int LOG2E_F_DEFAULT   = 22;
    localparam int LOG2E_W_DEFAULT   = LOG2E_I_DEFAULT + LOG2E_F_DEFAULT;  // 23
    // log2(e) = 1.44269504... in Q1.22
    localparam int LOG2E_VAL_DEFAULT = 'h5c551d;

    // =========================================================================
    // log2(e) product.
    //
    // The raw product is Q2.29 as in production; the difference is that it is
    // rounded to Q2.18 immediately and only the rounded value is carried
    // forward. 18 holds under RNE and half-up with the 64-entry table; the
    // sweep is not monotone (at 20 and at 17 the refit leaves one base-e
    // input wrong), so 18 is a verified point, not a bound.
    // =========================================================================
    localparam int MANT_MULT_I      = MANT_SRC_I + LOG2E_I_DEFAULT;      // 2
    localparam int MANT_MULT_FULL_F = MANT_SRC_F + LOG2E_F_DEFAULT;      // 29
    localparam int MANT_MULT_FULL_W = MANT_MULT_I + MANT_MULT_FULL_F;    // 31
    localparam int MANT_MULT_F      = 18;
    localparam int MANT_MULT_W      = MANT_MULT_I + MANT_MULT_F;         // 20

    // =========================================================================
    // Unified shift register.
    //
    // UNIFIED_F cannot go below MANT_MULT_F: bits present in the product would
    // be lost before the right shift can move them into range. It is therefore
    // exactly equal to it, which means FRAC_PAD = 0 -- unlike production, this
    // shifter really is bidirectional and a right shift really does drop bits.
    // The C++ model truncates identically (ac_fixed AC_TRN).
    // =========================================================================
    localparam int IN_CONV_INT_W    = INPUT_MAX_EXP + MANT_MULT_I;       // 9
    localparam int UNIFIED_I        = IN_CONV_INT_W;                     // 9
    localparam int UNIFIED_F        = MANT_MULT_F;                       // 18
    localparam int UNIFIED_W        = UNIFIED_I + UNIFIED_F;             // 27

    // =========================================================================
    // Polynomial input: top X_F fractional bits of the shifted value,
    // plain truncation.
    // =========================================================================
    localparam int X_F              = 16;
    localparam int X_W              = X_F;                               // 16

    // =========================================================================
    // Look-up table: 64 segments, indexed by the top 6 bits of x.
    // =========================================================================
    localparam int LUT_SIZE         = 64;
    localparam int LUT_ADDR_W       = 6;

    // =========================================================================
    // Coefficients. Neither ever reaches 1.0 (a in [0.348, 0.689], b in
    // [0.848, 0.99999]), so neither carries an integer bit. a needs far fewer
    // fractional bits than b: it only scales x in [0, 1/64) within a segment.
    // =========================================================================
    localparam int A_F              = 11;
    localparam int A_W              = A_F;                               // 11
    localparam int B_F              = 16;
    localparam int B_W              = B_F;                               // 16
    localparam int PACKED_W         = A_W + B_W;                         // 27

    // =========================================================================
    // Product and accumulator.
    //
    // Both multiplier operands are < 1, so a*x is too: Q0.27, and 11 x 16 fits
    // a single DSP48E1 with no cascade and no partial-product split.
    //
    // b - a*x lands in [0.5, 1) for every reachable input (the table fit
    // enforces it), so the accumulator is unsigned Q0.16 with no sign bit and
    // no integer guard bits.
    // =========================================================================
    localparam int PROD_F           = A_F + X_F;                         // 27
    localparam int PROD_W           = PROD_F;                            // 27
    localparam int CALC_F           = 16;
    localparam int CALC_W           = CALC_F;                            // 16
    // Bits of a*x dropped when aligning to the accumulator.
    localparam int CDROP            = PROD_F - CALC_F;                   // 11

    // =========================================================================
    // Normalisation.
    //
    // No priority encoder and no barrel shifter: b - a*x is known to be in
    // [0.5, 1), so the shift is either a constant 1 (STATIC_NORM) or one bit
    // test and a 2:1 mux covering [0.25, 1). Both produce the same Q1.16
    // result, so the guard costs no extra width.
    // =========================================================================
    localparam int POLY_OUT_I       = 1;
    localparam int POLY_OUT_F       = CALC_F;                            // 16
    localparam int POLY_OUT_W       = POLY_OUT_I + POLY_OUT_F;           // 17
    localparam int STATIC_POLY_EXP  = -1;

    // =========================================================================
    // Final round to BF16
    // =========================================================================
    localparam int BASE_SHIFT       = POLY_OUT_F - BF16_MANT_BITS;       // 9
    localparam int EXT_MANT_W       = BF16_MANT_BITS + 2;                // 9

endpackage : bf16_exp2_optim_pkg

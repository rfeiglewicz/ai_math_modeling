// =============================================================================
// bf16_exp2_pkg.sv
// SystemVerilog package with shared parameters for the BF16 exp2 pipeline.
// All bit-width constants are derived from the C++ model bf16_cfg namespace.
//
// Bit width summary (default parameters):
//   MANT_SRC_W  =  8  (1.7  unsigned)
//   LOG2E_W     = 23  (1.22 unsigned)
//   MANT_MULT_W = 31  (2.29 unsigned)
//   IN_F        = 38  (fractional bits from frac_part)
//   IN_W        = 39  (1.38 unsigned, used for poly input)
//   COEFF_W     = 21  (1.20 unsigned)
//   MULT_W      = 60  (2.58 unsigned)
//   CALC_W      = 62  (4.58 signed, with guard bit)
//   POLY_OUT_W  = 59  (1.58 unsigned)
//   BASE_SHIFT  = 51  (poly precision - target precision)
//   EXT_MANT_W  =  9  (7 + carry + hidden)
// =============================================================================

package bf16_exp2_pkg;

    // =========================================================================
    // BF16 Format Parameters
    // =========================================================================
    localparam int BF16_WIDTH       = 16;
    localparam int BF16_EXP_BITS    = 8;
    localparam int BF16_MANT_BITS   = 7;
    localparam int BF16_BIAS        = 127;
    localparam int BF16_MIN_EXP     = 1 - BF16_BIAS;  // -126

    // =========================================================================
    // Input Exponent Range for Approximation
    // =========================================================================
    localparam int INPUT_MIN_EXP    = -9;
    localparam int INPUT_MAX_EXP    = 7;

    // =========================================================================
    // Mantissa Source Format: 1.7 (hidden bit + 7 mantissa bits)
    // =========================================================================
    localparam int MANT_SRC_I       = 1;
    localparam int MANT_SRC_F       = BF16_MANT_BITS;  // 7
    localparam int MANT_SRC_W       = MANT_SRC_I + MANT_SRC_F;  // 8

    // =========================================================================
    // Default Log2(e) Constant Format: 1.22 -> 23 bits
    // Parameterizable per top-level
    // =========================================================================
    localparam int LOG2E_I_DEFAULT  = 1;
    localparam int LOG2E_F_DEFAULT  = 22;
    localparam int LOG2E_W_DEFAULT  = LOG2E_I_DEFAULT + LOG2E_F_DEFAULT;  // 23
    // log2(e) = 1.44269504... in fixed-point 1.22: value = 0x5c551d
    localparam int LOG2E_VAL_DEFAULT = 'h5c551d;

    // =========================================================================
    // Coefficient Format: 1.20 -> 21 bits each
    // Parameterizable per top-level
    // =========================================================================
    localparam int COEFF_I_DEFAULT  = 1;
    localparam int COEFF_F_DEFAULT  = 20;
    localparam int COEFF_W_DEFAULT  = COEFF_I_DEFAULT + COEFF_F_DEFAULT;  // 21
    localparam int COEFF_PACKED_W   = 2 * COEFF_W_DEFAULT;  // 42: [b|a]

    // =========================================================================
    // LUT Parameters
    // =========================================================================
    localparam int LUT_SIZE         = 128;
    localparam int LUT_ADDR_W       = 7;  // $clog2(128) = 7

    // =========================================================================
    // Core pipeline bit widths (derived, matches C++ bf16_cfg namespace)
    // =========================================================================
    localparam int MANT_MULT_I      = MANT_SRC_I + LOG2E_I_DEFAULT;  // 2
    localparam int MANT_MULT_F      = MANT_SRC_F + LOG2E_F_DEFAULT;  // 29
    localparam int MANT_MULT_W      = MANT_MULT_I + MANT_MULT_F;     // 31

    localparam int IN_I             = 1;
    localparam int IN_F             = MANT_MULT_F + (-INPUT_MIN_EXP); // 29+9=38
    localparam int IN_W             = IN_I + IN_F;                    // 39

    localparam int MULT_I           = IN_I + COEFF_I_DEFAULT;         // 2
    localparam int MULT_F           = IN_F + COEFF_F_DEFAULT;         // 58
    localparam int MULT_W           = MULT_I + MULT_F;                // 60

    localparam int CALC_I           = 4;  // MAX_OP_I(3) + GUARD(1)
    localparam int CALC_F           = MULT_F;                         // 58
    localparam int CALC_W           = CALC_I + CALC_F;                // 62

    localparam int POLY_OUT_I       = 1;
    localparam int POLY_OUT_F       = CALC_F;                         // 58
    localparam int POLY_OUT_W       = POLY_OUT_I + POLY_OUT_F;        // 59

    localparam int BASE_SHIFT       = POLY_OUT_F - BF16_MANT_BITS;   // 51
    localparam int EXT_MANT_W       = BF16_MANT_BITS + 2;            // 9

    // Width of unified fixed-point integer part (IN_CONV_INT_W from C++ model)
    localparam int IN_CONV_INT_W    = INPUT_MAX_EXP + MANT_MULT_I;   // 9

    // =========================================================================
    // Canonical early-out results.
    //
    // Every core must emit EXACTLY these patterns so that all implementations
    // are bit-identical for the whole 65536-value input space and can be
    // swapped for one another without touching anything downstream.
    //
    // BF16_QNAN is a fixed quiet NaN: the input NaN payload is NOT propagated.
    // Propagating it would make the result depend on the input mantissa, which
    // the table-based cores cannot reproduce without extra storage.
    // =========================================================================
    localparam logic [15:0] BF16_QNAN      = 16'hFFC0;
    localparam logic [15:0] BF16_PLUS_ONE  = 16'h3F80;
    localparam logic [15:0] BF16_PLUS_ZERO = 16'h0000;

    // =========================================================================
    // Common pipeline depth.
    //
    // The natural depths differ per core (4 for the table cores, 7 for exp2 and
    // poly4, 8 for poly4 with the DSP front end). Every core pads its output to
    // UNIFIED_PIPE_DEPTH so that latency and AXI-Stream timing are identical.
    // Set PIPE_TARGET=0 on a core to get its natural (unpadded) depth back.
    // =========================================================================
    localparam int UNIFIED_PIPE_DEPTH = 8;

    // =========================================================================
    // Decomposed BF16 status flags
    // =========================================================================
    typedef struct packed {
        logic is_nan;
        logic is_inf;
        logic is_denormal;
        logic is_zero;
    } fp_status_t;

    // =========================================================================
    // Decomposed BF16 structure
    // =========================================================================
    typedef struct packed {
        logic               sign;
        logic signed [8:0]  exponent;    // Unbiased, signed (range -126 to +127)
        logic [6:0]         mantissa;    // Explicit mantissa bits (no hidden bit)
        logic               hidden_bit;
        fp_status_t         status;
    } fp_raw_t;

    // =========================================================================
    // Early-out result codes
    // =========================================================================
    typedef enum logic [1:0] {
        EO_NONE      = 2'b00,  // No early out - use core result
        EO_PLUS_ONE  = 2'b01,  // Return +1.0
        EO_PLUS_ZERO = 2'b10,  // Return +0.0
        EO_QNAN      = 2'b11   // Return qNaN indefinite
    } early_out_t;

endpackage : bf16_exp2_pkg

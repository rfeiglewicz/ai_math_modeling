// =============================================================================
// bf16_unified_shift.sv
// Converts the mantissa fixed-point value (from log2e mux output) into the
// unified representation and applies the input exponent shift.
//
// Computes:
//   unified_val = mant_in << exponent   (if exponent >= 0)
//   unified_val = mant_in >> (-exponent) (if exponent < 0)
//
// Then separates the result into:
//   frac_part = lowest IN_F bits of unified_val (fractional part -> polynomial x)
//   int_part  = highest IN_CONV_INT_W bits -> -exponent_bias
//
// C++ model equivalent:
//   val = (unified_t)mant_val;
//   val <<= (or >>= ) exponent;
//   mant_val = val.slc<IN_F>(0);
//   exponent_bias = -(int)val.to_int();
//
// Bit widths (defaults match C++ namespace):
//   mant_in : MANT_MULT_W = 31 bits (2.29 format)
//   unified  : IN_CONV_INT_W + IN_F = 9 + 38 = 47 bits
//   frac_part: IN_F = 38 bits
//   int_part : IN_CONV_INT_W = 9 bits (signed, negated)
// =============================================================================

module bf16_unified_shift
    import bf16_exp2_pkg::*;
#(
    parameter bit REGISTER_OUTPUT = 1'b0,
    // 0 = bidirectional barrel shifter built from fabric MUX stages
    // 1 = one-hot multiply mapped onto DSP48 blocks (bit-exact, see below)
    parameter bit DSP_SHIFT       = 1'b0,
    parameter bit RESET_DATAPATH  = 1'b1,
    // Retiming: 1 inserts a register between the shift itself and the merge /
    // slice logic that follows it. In DSP mode that register lands in the DSP
    // PREG and stops the shift multiply and the coefficient multiply from
    // sharing one combinational stage. Bit-exact; costs one cycle of latency.
    parameter int EXTRA_STAGES    = 0
)(
    input  logic                                           clk,
    input  logic                                           rst_n,
    input  logic                                           pipe_en,
    input  logic [MANT_MULT_W-1:0]                         mant_in,    // 2.29 unsigned from log2e_mult
    input  logic signed [8:0]                              exponent,   // Unbiased input exponent
    output logic [IN_F-1:0]                                frac_part,  // 38-bit fractional (poly x)
    output logic signed [IN_CONV_INT_W-1:0]                int_part    // 9-bit integer (negated bias)
);

    // -------------------------------------------------------------------------
    // Place mant_in into unified 47-bit register at the correct position.
    // The unified format is (IN_CONV_INT_W.IN_F) = 9.38 = 47 bits total.
    // mant_in has format 2.29 = MANT_MULT_I.MANT_MULT_F.
    // To place it such that the binary point lines up with the unified format:
    //   frac extra bits = IN_F - MANT_MULT_F = 38 - 29 = 9 bits padding below
    //   int bits free   = IN_CONV_INT_W - MANT_MULT_I = 9 - 2 = 7 bits above
    // So the initial aligned position is: {7'b0, mant_in, 9'b0} in 47-bit field.
    // -------------------------------------------------------------------------
    localparam int UNIFIED_W   = IN_CONV_INT_W + IN_F;   // 47
    localparam int FRAC_PAD    = IN_F - MANT_MULT_F;     // 38 - 29 = 9
    localparam int INT_PAD     = IN_CONV_INT_W - MANT_MULT_I;  // 9 - 2 = 7

    // -------------------------------------------------------------------------
    // Apply input exponent
    // Note: exponent range [-9..7] so shift range already bounded by INPUT_MIN/MAX_EXP
    // -------------------------------------------------------------------------
    logic [UNIFIED_W-1:0] unified_shifted;

    generate
    if (!DSP_SHIFT) begin : gen_shift_barrel
        // ---------------------------------------------------------------------
        // Bidirectional barrel shifter: 5 stages of UNIFIED_W-wide 2:1 muxes
        // plus the left/right select => ~200 LUTs on 7-series.
        // ---------------------------------------------------------------------
        logic [UNIFIED_W-1:0] mant_aligned;
        logic [UNIFIED_W-1:0] shifted_comb;

        // mant_in placed at bits [FRAC_PAD + MANT_MULT_W - 1 : FRAC_PAD]
        assign mant_aligned = UNIFIED_W'({mant_in, {FRAC_PAD{1'b0}}});

        always_comb begin
            if ($signed(exponent) >= 0)
                shifted_comb = mant_aligned << exponent;
            else
                shifted_comb = mant_aligned >> (-exponent);
        end

        // Same optional retiming stage as the DSP branch, so that the pipeline
        // depth does not depend on which shifter is selected.
        if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_barrel_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)       unified_shifted <= '0;
                else if (pipe_en) unified_shifted <= shifted_comb;
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_barrel_reg
            always_ff @(posedge clk) begin
                if (pipe_en) unified_shifted <= shifted_comb;
            end
        end else begin : gen_barrel_wire
            assign unified_shifted = shifted_comb;
        end

    end else begin : gen_shift_dsp
        // ---------------------------------------------------------------------
        // Same function expressed as a single LEFT shift, then as a multiply by
        // a one-hot vector so it maps onto DSP48 blocks instead of fabric muxes.
        //
        // 1) Collapse the two shift directions into one.
        //    mant_aligned = mant_in << FRAC_PAD has FRAC_PAD (=9) zero LSBs, and
        //    -exponent never exceeds FRAC_PAD because INPUT_MIN_EXP = -FRAC_PAD.
        //    A right shift therefore never drops a set bit:
        //
        //        (mant_in << 9) >> k  ==  mant_in << (9 - k)     for 0 <= k <= 9
        //
        //    so for the whole in-range exponent span [-9, +7]:
        //
        //        unified_shifted = mant_in << s ,  s = exponent + FRAC_PAD
        //                                          s in [0, 16]
        //
        //    Out-of-range exponents are don't-care here: bf16_early_out forces
        //    EO_PLUS_ONE / EO_PLUS_ZERO and the top level muxes this path away.
        //
        // 2) A variable left shift is a multiply by a one-hot constant:
        //
        //        unified_shifted = mant_in * (1 << s)
        //
        // 3) mant_in is MANT_MULT_W (=31) bits, wider than the 25-bit DSP48 A
        //    port, so it is split at bit SPLIT_LO:
        //
        //        mant_in = mant_hi * 2^16 + mant_lo
        //        mant_in << s = (mant_hi << s) * 2^16 + (mant_lo << s)
        //
        //    mant_lo << s spans bits [s, s+15] and (mant_hi << s) << 16 spans
        //    bits [s+16, s+30]. The two ranges are disjoint for EVERY s, so the
        //    partial products merge with a bitwise OR - no carry, no adder.
        // ---------------------------------------------------------------------
        localparam int ONEHOT_W  = FRAC_PAD + INPUT_MAX_EXP + 1;   // 9 + 7 + 1 = 17
        localparam int SEL_W     = $clog2(ONEHOT_W);               // 5
        localparam int SPLIT_LO  = 16;                             // fits DSP B port
        localparam int SPLIT_HI  = MANT_MULT_W - SPLIT_LO;         // 31 - 16 = 15
        localparam int LO_PROD_W = SPLIT_LO + ONEHOT_W;            // 33
        localparam int HI_PROD_W = SPLIT_HI + ONEHOT_W;            // 32

        logic signed [9:0]   shift_sel;
        logic [ONEHOT_W-1:0] onehot;
        logic [SPLIT_LO-1:0] mant_lo;
        logic [SPLIT_HI-1:0] mant_hi;

        (* use_dsp = "yes" *) logic [LO_PROD_W-1:0] lo_prod;
        (* use_dsp = "yes" *) logic [HI_PROD_W-1:0] hi_prod;

        // s = exponent + FRAC_PAD ; 5-bit decode, out-of-range values are
        // don't-care (masked by early-out downstream).
        assign shift_sel = $signed({exponent[8], exponent}) + 10'sd9;
        assign onehot    = ONEHOT_W'(1'b1) << shift_sel[SEL_W-1:0];

        assign mant_lo = mant_in[SPLIT_LO-1 : 0];
        assign mant_hi = mant_in[MANT_MULT_W-1 : SPLIT_LO];

        assign lo_prod = mant_lo * onehot;   // DSP #1
        assign hi_prod = mant_hi * onehot;   // DSP #2

        // -------------------------------------------------------------------
        // Optional retiming register on the raw products.
        //
        // Without it the DSP result is merged, sliced and handed to the next
        // block's multiplier all in one stage, so two DSP multiplies end up in
        // series: about 3.8 ns in this DSP plus another 4 ns in the coefficient
        // multiply. The merge is combinational logic sitting between the DSP
        // and the output register, which is exactly what stops Vivado using
        // the DSP's own PREG. Registering the products first puts the register
        // back inside the block, where it costs no fabric.
        // -------------------------------------------------------------------
        logic [LO_PROD_W-1:0] lo_prod_s;
        logic [HI_PROD_W-1:0] hi_prod_s;

        if (EXTRA_STAGES >= 1 && RESET_DATAPATH) begin : gen_prod_reg_rst
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    lo_prod_s <= '0;
                    hi_prod_s <= '0;
                end else if (pipe_en) begin
                    lo_prod_s <= lo_prod;
                    hi_prod_s <= hi_prod;
                end
            end
        end else if (EXTRA_STAGES >= 1) begin : gen_prod_reg
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    lo_prod_s <= lo_prod;
                    hi_prod_s <= hi_prod;
                end
            end
        end else begin : gen_prod_wire
            assign lo_prod_s = lo_prod;
            assign hi_prod_s = hi_prod;
        end

        // Disjoint merge -- pure wiring plus SPLIT_LO..SPLIT_LO+15 OR gates.
        assign unified_shifted = (UNIFIED_W'(hi_prod_s) << SPLIT_LO) | UNIFIED_W'(lo_prod_s);
    end
    endgenerate

    // -------------------------------------------------------------------------
    // Extract fractional and integer parts
    // -------------------------------------------------------------------------
    logic [IN_F-1:0]               frac_part_comb;
    logic signed [IN_CONV_INT_W-1:0] int_part_comb;

    assign frac_part_comb = unified_shifted[IN_F-1:0];
    // exponent_bias = -(int)val.to_int() = negate integer part
    assign int_part_comb  = -signed'(unified_shifted[UNIFIED_W-1 : IN_F]);

    generate
        if (REGISTER_OUTPUT && RESET_DATAPATH) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    frac_part <= '0;
                    int_part  <= '0;
                end else if (pipe_en) begin
                    frac_part <= frac_part_comb;
                    int_part  <= int_part_comb;
                end
            end
        end else if (REGISTER_OUTPUT) begin : gen_reg_nrst
            always_ff @(posedge clk) begin
                if (pipe_en) begin
                    frac_part <= frac_part_comb;
                    int_part  <= int_part_comb;
                end
            end
        end else begin : gen_comb
            assign frac_part = frac_part_comb;
            assign int_part  = int_part_comb;
        end
    endgenerate

endmodule : bf16_unified_shift

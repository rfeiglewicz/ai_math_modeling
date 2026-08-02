#!/usr/bin/env bash
# =============================================================================
# pipe_depths.sh
#
# Measures the natural pipeline depth of every core variant used in the
# resource comparison. Nothing is computed here: each core reports its own
# depth at elaboration via -GREPORT_DEPTH=1, so the number always matches the
# RTL even if a retiming formula changes.
#
# Usage:  scripts/pipe_depths.sh
# =============================================================================
set -u
cd "$(dirname "$0")/.."

RTL=src/rtl
COMMON="$RTL/bf16_exp2_pkg.sv $RTL/bf16_pipe_pad.sv $RTL/bf16_decompose.sv $RTL/bf16_early_out.sv"

SRC_EXP2="$COMMON $RTL/bf16_recompose.sv $RTL/bf16_log2e_mult.sv \
          $RTL/bf16_unified_shift.sv $RTL/bf16_coeff_rom.sv \
          $RTL/bf16_linear_approx.sv $RTL/bf16_normalize.sv \
          $RTL/bf16_round.sv $RTL/bf16_exp2.sv"
SRC_LUT="$COMMON $RTL/bf16_expe_lut_rom.sv $RTL/bf16_expe_lut.sv"
SRC_HYB="$COMMON $RTL/bf16_expe_sparse_decode.sv $RTL/bf16_expe_hybrid_rom.sv $RTL/bf16_expe_hybrid.sv"
SRC_CUT="$COMMON $RTL/bf16_expe_cut_rom.sv $RTL/bf16_expe_cut.sv"
SRC_P4="$COMMON $RTL/bf16_expe_poly4_rom.sv $RTL/bf16_expe_poly4.sv"

RT_OFF_EXP2="-GRETIME_LOG2E=0 -GRETIME_SHIFT=0 -GRETIME_APPROX=0 -GRETIME_NORM=0 -GRETIME_ROUND=0 -GSPLIT_MULT=0"
RT_ON_EXP2="-GRETIME_LOG2E=2 -GRETIME_SHIFT=1 -GRETIME_APPROX=3 -GRETIME_NORM=1 -GRETIME_ROUND=2 -GSPLIT_MULT=1"
EXP2_OPT="-GDSP_SHIFT=1 -GMANT_MULT_ROUND_FRAC=21 -GRESET_DATAPATH=0"

probe () {
    local label="$1" top="$2" src="$3"; shift 3
    local out
    out=$(verilator --lint-only -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE \
          -Wno-LATCH -Wno-DECLFILENAME -I$RTL --top-module "$top" \
          -GREGISTER_STAGES=1 -GPIPE_TARGET=0 -GREPORT_DEPTH=1 "$@" \
          $src 2>&1 | grep -o 'CORE_PIPE_DEPTH core=[0-9]*' | head -1 | grep -o '[0-9]*$')
    printf '%-18s %s\n' "$label" "${out:-ERR}"
}

echo "PIPE_DEPTH_BEGIN"
printf '%-18s %s\n' "core" "stages"
probe "exp2 baseline"    bf16_exp2        "$SRC_EXP2" $RT_OFF_EXP2
probe "exp2 opt"         bf16_exp2        "$SRC_EXP2" $EXP2_OPT $RT_OFF_EXP2
probe "exp2 opt+retime"  bf16_exp2        "$SRC_EXP2" $EXP2_OPT $RT_ON_EXP2
probe "expe full-lut"    bf16_expe_lut    "$SRC_LUT"
probe "expe hybrid"      bf16_expe_hybrid "$SRC_HYB"
probe "expe cut"         bf16_expe_cut    "$SRC_CUT" -GRETIME_CUT=0 -GRETIME_FE=0
probe "expe cut+retime"  bf16_expe_cut    "$SRC_CUT" -GRETIME_CUT=2 -GRETIME_FE=1
probe "expe poly4"       bf16_expe_poly4  "$SRC_P4"  -GRETIME_MULT=0 -GRETIME_FE=0
probe "poly4+retime"     bf16_expe_poly4  "$SRC_P4"  -GRETIME_MULT=1 -GRETIME_FE=2
probe "poly4 dsp"        bf16_expe_poly4  "$SRC_P4"  -GDSP_FRONTEND=1 -GRESET_DATAPATH=0 -GRETIME_MULT=0 -GRETIME_FE=0
probe "poly4 dsp+retime" bf16_expe_poly4  "$SRC_P4"  -GDSP_FRONTEND=1 -GRESET_DATAPATH=0 -GRETIME_MULT=1 -GRETIME_FE=2
echo "PIPE_DEPTH_END"

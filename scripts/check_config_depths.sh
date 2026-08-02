#!/usr/bin/env bash
# Sprawdza, ze glebokosci zadeklarowane w sweep_retime_configs.tcl zgadzaja sie
# z tym, co RTL raportuje przy elaboracji. Uruchamiane recznie przy zmianie
# ktorejkolwiek formuly retimingu.
set -u
cd "$(dirname "$0")/.."
RTL=src/rtl
C="$RTL/bf16_exp2_pkg.sv $RTL/bf16_pipe_pad.sv $RTL/bf16_decompose.sv $RTL/bf16_early_out.sv"
CUT="$C $RTL/bf16_expe_cut_rom.sv $RTL/bf16_expe_cut.sv"
P4="$C $RTL/bf16_expe_poly4_rom.sv $RTL/bf16_expe_poly4.sv"

probe () {
    local top="$1" src="$2"; shift 2
    verilator --lint-only -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE \
        -Wno-LATCH -Wno-DECLFILENAME -I$RTL --top-module "$top" \
        -GREGISTER_STAGES=1 -GPIPE_TARGET=0 -GREPORT_DEPTH=1 "$@" $src 2>&1 \
        | grep -o 'core=[0-9]*' | head -1 | cut -d= -f2
}

for c in 0 1 2; do for f in 0 1; do
    printf 'cut C%s F%s   %s\n' "$c" "$f" "$(probe bf16_expe_cut "$CUT" -GRETIME_CUT=$c -GRETIME_FE=$f)"
done; done
for m in 0 1; do for f in 0 1 2; do
    printf 'poly4 M%s F%s %s\n' "$m" "$f" "$(probe bf16_expe_poly4 "$P4" -GRETIME_MULT=$m -GRETIME_FE=$f)"
done; done
for m in 0 1; do for f in 0 1 2; do
    printf 'p4dsp M%s F%s %s\n' "$m" "$f" "$(probe bf16_expe_poly4 "$P4" -GDSP_FRONTEND=1 -GRESET_DATAPATH=0 -GRETIME_MULT=$m -GRETIME_FE=$f)"
done; done

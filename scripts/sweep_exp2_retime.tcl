# Sweeps bf16_exp2 retiming configurations: how much Fmax does each extra
# pipeline register buy, and what does it cost in area?

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl        $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_compare
set part       xc7a200tfbg484-1
set period     2.0

file mkdir $out_dir
set xdc $out_dir/clk_fast.xdc
set fh [open $xdc w]
puts $fh "create_clock -period $period -name clk \[get_ports clk\]"
close $fh

set src {bf16_exp2_pkg.sv bf16_pipe_pad.sv bf16_decompose.sv bf16_early_out.sv
         bf16_recompose.sv bf16_log2e_mult.sv bf16_unified_shift.sv
         bf16_coeff_rom.sv bf16_linear_approx.sv bf16_normalize.sv
         bf16_round.sv bf16_exp2.sv}

set base "REGISTER_STAGES=1 DSP_SHIFT=1 MANT_MULT_ROUND_FRAC=21 RESET_DATAPATH=0 PIPE_TARGET=0"

# {label RETIME_LOG2E RETIME_APPROX RETIME_NORM RETIME_ROUND vivado_retiming}
set configs {
    {"L0 A0 N0 R0 base"  0 0 0 0 0}
    {"L0 A1 N1 R2"       0 1 1 2 0}
    {"L1 A1 N1 R2"       1 1 1 2 0}
    {"L1 A1 N1 R2 +retim" 1 1 1 2 1}
    {"L1 A2 N1 R2 +retim" 1 2 1 2 1}
}

set rows {}
foreach c $configs {
    lassign $c label rl ra rn rr rt

    close_project -quiet
    create_project -in_memory -part $part
    foreach f $src { read_verilog -sv $rtl/$f }
    read_xdc $xdc
    set gopt "$base RETIME_LOG2E=$rl RETIME_APPROX=$ra RETIME_NORM=$rn RETIME_ROUND=$rr"
    if {$rt} {
        synth_design -top bf16_exp2 -part $part -generic $gopt -retiming
    } else {
        synth_design -top bf16_exp2 -part $part -generic $gopt
    }

    set lut  [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]
    set srl  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ SRL*}]]
    set ff   [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    set dsp  [llength [get_cells -hierarchical -quiet -filter {REF_NAME == DSP48E1}]]

    set tp [lindex [get_timing_paths -max_paths 1 -nworst 1 -delay_type max] 0]
    set slack [get_property -quiet SLACK $tp]
    set fmax [format "%.1f" [expr {1000.0/($period-$slack)}]]
    set lvl  [get_property -quiet LOGIC_LEVELS $tp]
    set ep   [get_property -quiet ENDPOINT_PIN $tp]

    lappend rows [list $label [expr {$lut-$srl}] $ff $dsp $fmax $lvl \
                       [expr {7+$rl+$ra+$rn+$rr}] $ep]
}

puts ""
puts "RETIME_SWEEP_BEGIN"
puts [format "%-18s %9s %6s %5s %10s %6s %6s  %s" cfg LUT_logic FF DSP Fmax_MHz lvls depth endpoint]
puts [string repeat "-" 120]
foreach r $rows {
    lassign $r label lut ff dsp fmax lvl dep ep
    puts [format "%-18s %9d %6d %5d %10s %6s %6s  %s" $label $lut $ff $dsp $fmax $lvl $dep $ep]
}
puts "RETIME_SWEEP_END"

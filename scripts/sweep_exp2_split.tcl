# Does splitting the coefficient multiply into per-DSP partial products break
# the DSP-to-DSP cascade that was capping bf16_exp2?

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

set base "REGISTER_STAGES=1 DSP_SHIFT=1 RESET_DATAPATH=0 PIPE_TARGET=0 MANT_MULT_ROUND_FRAC=21"
set rt   "RETIME_NORM=1 RETIME_ROUND=2"

# {label RETIME_LOG2E RETIME_SHIFT RETIME_APPROX SPLIT_MULT}
set configs {
    {"L1 S0 A1 sp0"  1 0 1 0}
    {"L2 S1 A2 sp1"  2 1 2 1}
    {"L2 S1 A3 sp1"  2 1 3 1}
    {"L1 S1 A3 sp1"  1 1 3 1}
}

set rows {}
foreach c $configs {
    lassign $c label rl rs ra sp

    close_project -quiet
    create_project -in_memory -part $part
    foreach f $src { read_verilog -sv $rtl/$f }
    read_xdc $xdc
    synth_design -top bf16_exp2 -part $part -generic \
      "$base $rt RETIME_LOG2E=$rl RETIME_SHIFT=$rs RETIME_APPROX=$ra SPLIT_MULT=$sp"

    set lut  [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]
    set srl  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ SRL*}]]
    set ff   [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    set dsp  [llength [get_cells -hierarchical -quiet -filter {REF_NAME == DSP48E1}]]

    set tp [lindex [get_timing_paths -max_paths 1 -nworst 1 -delay_type max] 0]
    set slack [get_property -quiet SLACK $tp]
    set fmax [format "%.1f" [expr {1000.0/($period-$slack)}]]
    set lvl  [get_property -quiet LOGIC_LEVELS $tp]
    set ep   [get_property -quiet ENDPOINT_PIN $tp]

    set tag [string map {" " _} $label]
    report_timing -max_paths 3 -unique_pins -delay_type max \
                  -file $out_dir/crit_split_${tag}.rpt

    lappend rows [list $label [expr {$lut-$srl}] $ff $dsp $fmax $lvl \
                       [expr {7+$rl+$rs+$ra+1+2}] $ep]
}

puts ""
puts "SPLIT_SWEEP_BEGIN"
puts [format "%-18s %9s %6s %5s %10s %6s %6s  %s" cfg LUT_logic FF DSP Fmax_MHz lvls depth endpoint]
puts [string repeat "-" 120]
foreach r $rows {
    lassign $r label lut ff dsp fmax lvl dep ep
    puts [format "%-18s %9d %6d %5d %10s %6s %6s  %s" $label $lut $ff $dsp $fmax $lvl $dep $ep]
}
puts "SPLIT_SWEEP_END"

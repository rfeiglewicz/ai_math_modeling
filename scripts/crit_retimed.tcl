# Where is the critical path AFTER the linear-approx / normalize split?

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

set gen [expr {[info exists ::env(CRIT_GEN)] ? $::env(CRIT_GEN) : \
  "RETIME_APPROX=1 RETIME_NORM=1 RETIME_ROUND=2"}]

close_project -quiet
create_project -in_memory -part $part
foreach f $src { read_verilog -sv $rtl/$f }
read_xdc $xdc
synth_design -top bf16_exp2 -part $part -generic \
  "REGISTER_STAGES=1 DSP_SHIFT=1 MANT_MULT_ROUND_FRAC=21 RESET_DATAPATH=0 PIPE_TARGET=0 $gen"

report_timing -max_paths 8 -unique_pins -delay_type max -file $out_dir/crit_retimed.rpt

puts ""
puts "TOP_PATHS_BEGIN"
set i 0
foreach tp [get_timing_paths -max_paths 8 -unique_pins -delay_type max] {
    incr i
    puts [format "%2d  slack %8s  lvls %3s" $i \
          [get_property -quiet SLACK $tp] [get_property -quiet LOGIC_LEVELS $tp]]
    puts "      from [get_property -quiet STARTPOINT_PIN $tp]"
    puts "      to   [get_property -quiet ENDPOINT_PIN   $tp]"
}
puts "TOP_PATHS_END"

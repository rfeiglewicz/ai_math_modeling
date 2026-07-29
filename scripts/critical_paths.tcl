# Dumps the critical path of each core so we can see WHERE the time goes
# instead of guessing.
#
#   CRIT_ONLY - optional list of core labels to restrict the run

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl        $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_compare
set part       xc7a200tfbg484-1
set period     4.0
set only [expr {[info exists ::env(CRIT_ONLY)] ? $::env(CRIT_ONLY) : ""}]

file mkdir $out_dir
set xdc $out_dir/clk.xdc
set fh [open $xdc w]
puts $fh "create_clock -period $period -name clk \[get_ports clk\]"
close $fh

set common {bf16_exp2_pkg.sv bf16_pipe_pad.sv bf16_decompose.sv bf16_early_out.sv}
set src_exp2  [concat $common {bf16_recompose.sv bf16_log2e_mult.sv \
                               bf16_unified_shift.sv bf16_coeff_rom.sv \
                               bf16_linear_approx.sv bf16_normalize.sv \
                               bf16_round.sv bf16_exp2.sv}]
set src_lut    [concat $common {bf16_expe_lut_rom.sv bf16_expe_lut.sv}]
set src_hybrid [concat $common {bf16_expe_sparse_decode.sv bf16_expe_hybrid_rom.sv \
                                bf16_expe_hybrid.sv}]
set src_cut    [concat $common {bf16_expe_cut_rom.sv bf16_expe_cut.sv}]
set src_poly4  [concat $common {bf16_expe_poly4_rom.sv bf16_expe_poly4.sv}]

set variants [list \
  [list "exp2 optimised"  bf16_exp2        $src_exp2   "REGISTER_STAGES=1 DSP_SHIFT=1 MANT_MULT_ROUND_FRAC=21 RESET_DATAPATH=0"] \
  [list "expe full-lut"   bf16_expe_lut    $src_lut    "REGISTER_STAGES=1"] \
  [list "expe hybrid"     bf16_expe_hybrid $src_hybrid "REGISTER_STAGES=1"] \
  [list "expe cut-ladder" bf16_expe_cut    $src_cut    "REGISTER_STAGES=1"] \
  [list "expe poly4"      bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1"] \
  [list "expe poly4 dsp"  bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1 DSP_FRONTEND=1 RESET_DATAPATH=0"] \
]

foreach v $variants {
    lassign $v label top files generics
    if {$only ne "" && [lsearch -exact $only $label] < 0} { continue }

    close_project -quiet
    create_project -in_memory -part $part
    foreach f $files { read_verilog -sv $rtl/$f }
    read_xdc $xdc
    synth_design -top $top -part $part -generic $generics

    set tag [string map {" " _} $label]
    report_timing -max_paths 3 -nworst 3 -delay_type max -input_pins \
                  -file $out_dir/crit_${tag}.rpt

    set tp [lindex [get_timing_paths -max_paths 1 -nworst 1 -delay_type max] 0]
    puts ""
    puts "CRIT $label"
    puts "  slack   [get_property -quiet SLACK $tp]"
    puts "  delay   [get_property -quiet DATAPATH_DELAY $tp]"
    puts "  logic_lv[get_property -quiet LOGIC_LEVELS $tp]"
    puts "  from    [get_property -quiet STARTPOINT_PIN $tp]"
    puts "  to      [get_property -quiet ENDPOINT_PIN $tp]"
}
puts "CRIT_DONE"

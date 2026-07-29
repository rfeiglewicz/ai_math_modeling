# Sanity probe: does constraining synthesis actually change the result?
# Runs the same core twice - once unconstrained (clock created after
# synth_design, as compare_all_cores.tcl used to do) and once properly
# constrained via an XDC read before synth_design.

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl        $repo_root/src/rtl
set part       xc7a200tfbg484-1
set period     4.0

set src {bf16_exp2_pkg.sv bf16_pipe_pad.sv bf16_decompose.sv bf16_early_out.sv
         bf16_recompose.sv bf16_log2e_mult.sv bf16_unified_shift.sv
         bf16_coeff_rom.sv bf16_linear_approx.sv bf16_normalize.sv
         bf16_round.sv bf16_exp2.sv}

set generics "REGISTER_STAGES=1 DSP_SHIFT=1 MANT_MULT_ROUND_FRAC=21 RESET_DATAPATH=0"

set xdc $repo_root/build/vivado_compare/probe.xdc
file mkdir [file dirname $xdc]
set fh [open $xdc w]
puts $fh "create_clock -period $period -name clk \[get_ports clk\]"
close $fh

proc measure {period} {
    set fmax "n/a"
    set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $tp] > 0} {
        set slack [get_property -quiet SLACK $tp]
        if {$slack ne ""} { set fmax [format "%.1f" [expr {1000.0/($period-$slack)}]] }
    }
    return $fmax
}

# --- A: unconstrained during synthesis (the old flow) ---
close_project -quiet
create_project -in_memory -part $part
foreach f $src { read_verilog -sv $rtl/$f }
synth_design -top bf16_exp2 -part $part -generic $generics
create_clock -period $period -name clk [get_ports clk]
set fmax_a [measure $period]
set lut_a  [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]

# --- B: constrained during synthesis ---
close_project -quiet
create_project -in_memory -part $part
foreach f $src { read_verilog -sv $rtl/$f }
read_xdc $xdc
synth_design -top bf16_exp2 -part $part -generic $generics
set fmax_b [measure $period]
set lut_b  [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]

puts ""
puts "PROBE_BEGIN"
puts [format "%-34s %10s %8s" flow Fmax_MHz LUT]
puts [format "%-34s %10s %8d" "A: clock after synth (old flow)"  $fmax_a $lut_a]
puts [format "%-34s %10s %8d" "B: XDC read before synth"         $fmax_b $lut_b]
puts "PROBE_END"

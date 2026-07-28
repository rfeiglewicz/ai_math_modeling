# Synthesizes every interesting configuration of bf16_expe_poly4 and prints one
# comparison table.  This is the evidence behind the resource claims: no number
# in the summary is estimated, all of them come from synth_design on the same
# part with the same clock constraint.
#
#   vivado -mode batch -source scripts/sweep_poly4_configs.tcl

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl_dir    [file join $repo_root src rtl]
set out_dir    [file join $repo_root build vivado_poly4_sweep]
file mkdir $out_dir

set part   xc7a200tfbg484-1
set period 4.0

set sources [list \
    [file join $rtl_dir bf16_exp2_pkg.sv] \
    [file join $rtl_dir bf16_decompose.sv] \
    [file join $rtl_dir bf16_early_out.sv] \
    [file join $rtl_dir bf16_expe_poly4_rom.sv] \
    [file join $rtl_dir bf16_expe_poly4.sv]]

# label  DSP_FRONTEND  RESET_DATAPATH
set configs {
    {"barrel + async reset (baseline)" 0 1}
    {"barrel + no dp reset"            0 0}
    {"DSP front end + async reset"     1 1}
    {"DSP front end + no dp reset"     1 0}
}

set rows {}

foreach cfg $configs {
    lassign $cfg label fe rst

    close_design -quiet
    read_verilog -sv $sources
    synth_design -top bf16_expe_poly4 -part $part \
        -generic "REGISTER_STAGES=1 DSP_FRONTEND=$fe RESET_DATAPATH=$rst"

    create_clock -period $period -name clk [get_ports clk]

    set tag "fe${fe}_rst${rst}"
    report_utilization -file [file join $out_dir "util_$tag.rpt"]
    report_timing_summary -delay_type max -max_paths 1 \
        -file [file join $out_dir "timing_$tag.rpt"]

    set luts  [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT}]]
    set srls  [llength [get_cells -hierarchical -filter {REF_NAME =~ SRL*}]]
    set carry [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == CARRY}]]
    set ffs   [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    set dsps  [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]
    set brams [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == BLOCKRAM}]]

    # Worst setup slack on the internal paths -> achievable clock period.
    set slack 0.0
    set paths [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
    if {[llength $paths] > 0} {
        set slack [get_property SLACK [lindex $paths 0]]
    }
    if {$slack eq "" || $slack eq "inf"} {
        set fmax "n/a"
    } else {
        set fmax [format "%.0f" [expr {1000.0 / ($period - $slack)}]]
    }

    lappend rows [list $label $luts $srls $carry $ffs $dsps $brams $fmax]
}

puts ""
puts "RESULTS_TABLE_BEGIN"
puts [format "%-32s %6s %6s %6s %6s %5s %6s %8s" \
        "configuration" "LUT" "SRL" "CARRY" "FF" "DSP" "BRAM" "Fmax MHz"]
foreach r $rows {
    lassign $r label luts srls carry ffs dsps brams fmax
    puts [format "%-32s %6d %6d %6d %6d %5d %6d %8s" \
            $label $luts $srls $carry $ffs $dsps $brams $fmax]
}
puts "RESULTS_TABLE_END"
puts "reports in $out_dir"

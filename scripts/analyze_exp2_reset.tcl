# Does the asynchronous reset cost LUTs in bf16_exp2 the way it did in
# bf16_expe_poly4?  Synthesizes the module and reports where the LUTs go, plus
# how many LUTs sit directly on a DSP or BRAM input pin (that is the signature
# of reset/enable emulation around a dedicated block).

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl_dir    [file join $repo_root src rtl]
set out_dir    [file join $repo_root build vivado_exp2_reset]
file mkdir $out_dir

read_verilog -sv [list \
    [file join $rtl_dir bf16_exp2_pkg.sv] \
    [file join $rtl_dir bf16_decompose.sv] \
    [file join $rtl_dir bf16_recompose.sv] \
    [file join $rtl_dir bf16_early_out.sv] \
    [file join $rtl_dir bf16_log2e_mult.sv] \
    [file join $rtl_dir bf16_unified_shift.sv] \
    [file join $rtl_dir bf16_coeff_rom.sv] \
    [file join $rtl_dir bf16_linear_approx.sv] \
    [file join $rtl_dir bf16_normalize.sv] \
    [file join $rtl_dir bf16_round.sv] \
    [file join $rtl_dir bf16_exp2.sv]]

synth_design -top bf16_exp2 -part xc7a200tfbg484-1 \
    -generic {REGISTER_STAGES=1}

report_utilization -file [file join $out_dir util_exp2.rpt]
report_utilization -hierarchical -file [file join $out_dir util_exp2_hier.rpt]

puts ""
puts "EXP2_ANALYSIS_BEGIN"
puts "LUT   [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT}]]"
puts "CARRY [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == CARRY}]]"
puts "FF    [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]"
puts "DSP   [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]"
puts "BRAM  [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == BLOCKRAM}]]"

# How many flops sit in fabric with an asynchronous clear?  FDCE/FDPE means
# async, FDRE/FDSE means synchronous (or no) reset.
foreach ref {FDCE FDPE FDRE FDSE} {
    puts "$ref [llength [get_cells -hierarchical -filter "REF_NAME == $ref"]]"
}

# LUTs whose output drives a DSP or BRAM input: reset/enable emulation.
set gate_dsp  0
set gate_bram 0
foreach c [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT}] {
    set onet [get_nets -quiet -of_objects [get_pins -quiet -of_objects $c -filter {DIRECTION == OUT}]]
    if {[llength $onet] == 0} { continue }
    set loads [get_cells -quiet -of_objects [get_pins -quiet -of_objects $onet -filter {DIRECTION == IN}]]
    foreach l $loads {
        set r [get_property REF_NAME $l]
        if {$r eq "DSP48E1"} { incr gate_dsp; break }
        if {[string match "RAMB*" $r]} { incr gate_bram; break }
    }
}
puts "LUTS_DRIVING_DSP  $gate_dsp"
puts "LUTS_DRIVING_BRAM $gate_bram"

array set bucket {}
foreach c [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT || PRIMITIVE_GROUP == CARRY}] {
    set key [get_property NAME $c]
    regsub {\[[0-9]+\]$} $key {[*]} key
    regsub {_[0-9]+$} $key {_N} key
    regsub {_i_[0-9]+} $key {_i_N} key
    incr bucket($key)
}
puts "--- LUT/CARRY by signal (only groups of 4 or more) ---"
foreach k [lsort [array names bucket]] {
    if {$bucket($k) >= 4} { puts [format "%5d  %s" $bucket($k) $k] }
}
puts "EXP2_ANALYSIS_END"

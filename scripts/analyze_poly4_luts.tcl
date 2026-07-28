# Where do the LUTs go?  Groups every LUT/CARRY cell by its RTL parent so that
# the LUT budget can be attacked at the right place instead of by guesswork.

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl_dir    [file join $repo_root src rtl]

if {![info exists ::env(POLY4_DSP_FRONTEND)]} {
    set fe 1
} else {
    set fe $::env(POLY4_DSP_FRONTEND)
}

read_verilog -sv [list \
    [file join $rtl_dir bf16_exp2_pkg.sv] \
    [file join $rtl_dir bf16_decompose.sv] \
    [file join $rtl_dir bf16_early_out.sv] \
    [file join $rtl_dir bf16_expe_poly4_rom.sv] \
    [file join $rtl_dir bf16_expe_poly4.sv]]

synth_design -top bf16_expe_poly4 -part xc7a200tfbg484-1 \
    -generic "REGISTER_STAGES=1 DSP_FRONTEND=$fe"

# Strip the trailing bit index so that a 25-bit vector of LUTs collapses to one
# line; that is what identifies the guilty RTL expression.
array set bucket {}
foreach c [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT || PRIMITIVE_GROUP == CARRY}] {
    set key [get_property NAME $c]
    regsub {\[[0-9]+\]$} $key {[*]} key
    regsub {_[0-9]+$} $key {_N} key
    regsub {_i_[0-9]+} $key {_i_N} key
    incr bucket($key)
}

puts "=== LUT/CARRY cells by signal (DSP_FRONTEND=$fe) ==="
foreach k [lsort [array names bucket]] {
    puts [format "%5d  %s" $bucket($k) $k]
}
puts "=== totals ==="
puts "LUT   [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == LUT}]]"
puts "CARRY [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == CARRY}]]"
puts "DSP   [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]]"
puts "FF    [llength [get_cells -hierarchical -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]"

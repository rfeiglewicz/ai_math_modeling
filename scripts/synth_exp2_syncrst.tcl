# Kontrolna synteza bf16_exp2 z wariantu, w ktorym reset asynchroniczny
# zostal zamieniony na synchroniczny (kopia RTL w build/exp2_syncrst).
set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set src_dir    $repo_root/build/exp2_syncrst
set out_dir    $repo_root/build/vivado_exp2_syncrst

file mkdir $out_dir

set files {
    bf16_exp2_pkg.sv
    bf16_decompose.sv
    bf16_recompose.sv
    bf16_early_out.sv
    bf16_log2e_mult.sv
    bf16_unified_shift.sv
    bf16_coeff_rom.sv
    bf16_linear_approx.sv
    bf16_normalize.sv
    bf16_round.sv
    bf16_exp2.sv
}
foreach f $files {
    read_verilog -sv $src_dir/$f
}

synth_design -top bf16_exp2 -part xc7a200tfbg484-1 -generic {REGISTER_STAGES=1}

puts "EXP2_SYNC_BEGIN"
puts "LUT   [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]"
puts "CARRY [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == CARRY}]]"
puts "FF    [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]"
puts "DSP   [llength [get_cells -hierarchical -quiet -filter {REF_NAME == DSP48E1}]]"
foreach ref {FDCE FDPE FDRE FDSE} {
    puts "$ref [llength [get_cells -hierarchical -quiet -filter "REF_NAME == $ref"]]"
}

set gate_dsp 0
foreach c [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}] {
    set onet [get_nets -quiet -of_objects [get_pins -quiet -of_objects $c -filter {DIRECTION == OUT}]]
    if {[llength $onet] == 0} { continue }
    foreach p [get_pins -quiet -of_objects $onet -filter {DIRECTION == IN}] {
        set r [get_property -quiet REF_NAME [get_cells -quiet -of_objects $p]]
        if {$r eq "DSP48E1"} { incr gate_dsp; break }
    }
}
puts "LUTS_DRIVING_DSP  $gate_dsp"
puts "EXP2_SYNC_END"

report_utilization -file $out_dir/utilization.rpt

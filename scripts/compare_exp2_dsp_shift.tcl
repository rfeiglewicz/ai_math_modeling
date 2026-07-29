# Porownanie bf16_exp2: barrel shifter w fabricu vs mnozenie one-hot na DSP.
# Kazdy wariant syntezowany osobno, wynik w tabeli.

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl_dir    $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_exp2_dspshift
set part       xc7a200tfbg484-1

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

# nazwa  DSP_SHIFT
set configs {
    {barrel 0}
    {dsp    1}
}

set results {}

foreach cfg $configs {
    lassign $cfg name dsp_shift

    close_project -quiet
    create_project -in_memory -part $part

    foreach f $files {
        read_verilog -sv $rtl_dir/$f
    }

    synth_design -top bf16_exp2 -part $part \
        -generic "REGISTER_STAGES=1 DSP_SHIFT=$dsp_shift"

    create_clock -period 4.000 -name clk [get_ports clk]

    set lut   [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]
    set carry [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == CARRY}]]
    set ff    [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    set dsp   [llength [get_cells -hierarchical -quiet -filter {REF_NAME == DSP48E1}]]
    set bram  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB*}]]

    # LUT-y nalezace do modulu przesuwnika
    set shift_lut [llength [get_cells -hierarchical -quiet -filter \
        {PRIMITIVE_GROUP == LUT && NAME =~ *u_unified_shift*}]]

    set wns "n/a"
    set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $tp] > 0} {
        set slack [get_property -quiet SLACK $tp]
        if {$slack ne ""} {
            set wns [format "%.3f" $slack]
        }
    }

    report_utilization -hierarchical -file $out_dir/util_${name}_hier.rpt
    report_utilization -file $out_dir/util_${name}.rpt

    lappend results [list $name $lut $carry $ff $dsp $bram $shift_lut $wns]
}

puts "DSPSHIFT_TABLE_BEGIN"
puts [format "%-8s %8s %8s %8s %6s %6s %12s %10s" \
      config LUT CARRY FF DSP BRAM shift_LUT WNS_ns]
foreach r $results {
    lassign $r name lut carry ff dsp bram shift_lut wns
    puts [format "%-8s %8d %8d %8d %6d %6d %12d %10s" \
          $name $lut $carry $ff $dsp $bram $shift_lut $wns]
}
puts "DSPSHIFT_TABLE_END"
puts "REPORT_DIR $out_dir"

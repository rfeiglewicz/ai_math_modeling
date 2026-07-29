# Przemiatanie konfiguracji bf16_exp2:
#   - DSP_SHIFT            : barrel shifter w fabricu vs mnozenie one-hot na DSP
#   - MANT_MULT_ROUND_FRAC : szerokosc iloczynu log2(e) po zaokragleniu RNE
#   - RESET_DATAPATH       : asynchroniczny reset na rejestrach sciezki danych
#
# Kazdy wariant syntezowany od zera; liczby pochodza z synth_design.

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl_dir    $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_exp2_sweep
set part       xc7a200tfbg484-1
set period     4.000

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

# nazwa            DSP_SHIFT  ROUND_FRAC  RESET_DATAPATH
set configs {
    {baseline           0   29   1}
    {dsp                1   29   1}
    {dsp+rf21           1   21   1}
    {dsp+rf21+nrst      1   21   0}
}

set results {}

foreach cfg $configs {
    lassign $cfg name dsp_shift round_frac rst_dp

    close_project -quiet
    create_project -in_memory -part $part

    foreach f $files {
        read_verilog -sv $rtl_dir/$f
    }

    synth_design -top bf16_exp2 -part $part -generic \
        "REGISTER_STAGES=1 DSP_SHIFT=$dsp_shift MANT_MULT_ROUND_FRAC=$round_frac RESET_DATAPATH=$rst_dp"

    create_clock -period $period -name clk [get_ports clk]

    set lut   [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]
    set carry [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == CARRY}]]
    set ff    [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    set dsp   [llength [get_cells -hierarchical -quiet -filter {REF_NAME == DSP48E1}]]
    set bram  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB*}]]

    set fmax "n/a"
    set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $tp] > 0} {
        set slack [get_property -quiet SLACK $tp]
        if {$slack ne "" && $slack < $period} {
            set fmax [format "%.1f" [expr {1000.0 / ($period - $slack)}]]
        }
    }

    report_utilization -hierarchical -file $out_dir/util_${name}_hier.rpt
    report_utilization -file $out_dir/util_${name}.rpt

    lappend results [list $name $lut $carry $ff $dsp $bram $fmax]
}

puts "EXP2_SWEEP_BEGIN"
puts [format "%-16s %7s %7s %7s %5s %5s %9s" config LUT CARRY FF DSP BRAM Fmax_MHz]
foreach r $results {
    lassign $r name lut carry ff dsp bram fmax
    puts [format "%-16s %7d %7d %7d %5d %5d %9s" $name $lut $carry $ff $dsp $bram $fmax]
}
puts "EXP2_SWEEP_END"
puts "REPORT_DIR $out_dir"

# =============================================================================
# sweep_optim_configs.tcl
#
# Przemiatanie konfiguracji bf16_exp2_optim -- rdzenia o zwezonej sciezce
# danych. Odpowiada na trzy pytania, ktore w kodzie sa zapisane jako tezy, a
# tu zostaja zmierzone:
#
#   ROUND_MODE      czy zaokraglanie half-up jest tansze od RNE. Po
#                   zmaksymalizowaniu retimingu sciezka krytyczna calego
#                   rdzenia siedzi wlasnie w sumatorze zaokraglajacym iloczyn
#                   log2(e), wiec to nie jest pytanie akademickie.
#                   Oba tryby schodza do 21 bitow (sprawdzone wyczerpujaco
#                   przez make rtl_optim_round_frac_sweep), wiec wybor jest
#                   darmowy funkcjonalnie.
#
#   STATIC_NORM     ile naprawde kosztuje strzezona normalizacja. Komentarz w
#                   RTL mowi, ze nie kosztuje dodatkowej SZEROKOSCI -- to
#                   prawda i wynika z zakresow -- ale szerokosc to nie to samo
#                   co LUT-y.
#
#   RESET_DATAPATH  ile kosztuje asynchroniczny reset na sciezce danych, czyli
#                   ile Vivado zyskuje mogac wciagnac rejestry do DSP48/BRAM.
#
# Uruchomienie:
#   make sweep_optim
#
# Zmienne srodowiskowe:
#   OPTIM_PART   - uklad docelowy    (domyslnie xc7a200tfbg484-1)
#   OPTIM_PERIOD - okres zegara w ns (domyslnie 2.0, celowo nieosiagalny)
# =============================================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl        $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_optim_sweep

set part   [expr {[info exists ::env(OPTIM_PART)]   ? $::env(OPTIM_PART)   : "xc7a200tfbg484-1"}]
# Tak samo jak w compare_all_cores.tcl: ograniczenie musi byc nieosiagalne,
# inaczej synteza przestaje optymalizowac po jego spelnieniu i Fmax wychodzi
# zanizone.
set period [expr {[info exists ::env(OPTIM_PERIOD)] ? $::env(OPTIM_PERIOD) : 2.0}]

file mkdir $out_dir

# Zegar musi istniec PRZED synth_design, inaczej caly przebieg jest
# nieograniczony czasowo.
set xdc $out_dir/clk.xdc
set fh [open $xdc w]
puts $fh "create_clock -period $period -name clk \[get_ports clk\]"
close $fh

set files {
    bf16_exp2_pkg.sv
    bf16_exp2_optim_pkg.sv
    bf16_pipe_pad.sv
    bf16_decompose.sv
    bf16_recompose.sv
    bf16_early_out.sv
    bf16_exp2_optim_rom.sv
    bf16_exp2_optim_log2e_mult.sv
    bf16_exp2_optim_shift.sv
    bf16_exp2_optim_approx.sv
    bf16_exp2_optim_normalize.sv
    bf16_exp2_optim_round.sv
    bf16_exp2_optim.sv
}

# Retiming do oporu we wszystkich wariantach, zeby porownanie dotyczylo
# badanej opcji, a nie tego, ktory wariant dostal wiecej stopni.
set RT_MAX "RETIME_LOG2E=2 RETIME_SHIFT=1 RETIME_APPROX=2 RETIME_NORM=1 RETIME_ROUND=2"

# nazwa          ROUND_MODE  STATIC_NORM  RESET_DATAPATH
set configs {
    {rne                0   0   0}
    {half-up            1   0   0}
    {rne+static         0   1   0}
    {half-up+static     1   1   0}
    {rne+reset          0   0   1}
}

set results {}

proc util_row {txt name idx} {
    set pat "\\|\\s+[string map {* {\\*}} $name]\\s*\\|(\[^\n\]*)\\|"
    if {[regexp $pat $txt -> rest]} {
        set cols [split $rest "|"]
        set v [string trim [lindex $cols $idx]]
        if {[string is double -strict $v]} { return $v }
    }
    return 0
}

foreach cfg $configs {
    lassign $cfg name round_mode static_norm rst_dp

    puts "=== syntezuje: $name ==="

    close_project -quiet
    create_project -in_memory -part $part

    foreach f $files { read_verilog -sv $rtl/$f }
    read_xdc $xdc

    synth_design -top bf16_exp2_optim -part $part -generic \
        "REGISTER_STAGES=1 PIPE_TARGET=0 ROUND_MODE=$round_mode \
         STATIC_NORM=$static_norm RESET_DATAPATH=$rst_dp $RT_MAX"

    # Zasoby z report_utilization (miejsca w ukladzie), nie z get_cells
    # (prymitywy) -- dwa LUT5 dziela jedno miejsce LUT6.
    set util [report_utilization -return_string]

    set lut   [util_row $util "LUT as Logic"           0]
    set srl   [util_row $util "LUT as Shift Register"  0]
    set ff    [util_row $util "Register as Flip Flop"  0]
    set dsp   [util_row $util "DSPs"                   0]
    set bram  [util_row $util "Block RAM Tile"         0]
    set carry [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == CARRY}]]

    set fmax "n/a"
    set lvls "-"
    set src  "-"
    set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $tp] > 0} {
        set slack [get_property -quiet SLACK $tp]
        if {$slack ne ""} {
            set fmax [format "%.1f" [expr {1000.0 / ($period - $slack)}]]
            set lvls [get_property -quiet LOGIC_LEVELS $tp]
            # Skad startuje sciezka krytyczna: to mowi, ktory blok jest
            # limitem, a nie tylko ze jakis jest.
            set sp [get_property -quiet STARTPOINT_PIN $tp]
            if {$sp ne ""} {
                set src [lindex [split [get_property -quiet NAME $sp] /] 0]
            }
        }
    }

    report_utilization -file $out_dir/util_${name}.rpt
    report_timing -max_paths 3 -unique_pins -delay_type max \
                  -file $out_dir/crit_${name}.rpt

    lappend results [list $name $lut $carry $ff $srl $dsp $bram $fmax $lvls $src]
}

puts ""
puts "OPTIM_SWEEP_BEGIN"
puts "part   $part"
puts "period ${period} ns"
puts ""
puts [format "%-16s %9s %7s %6s %5s %5s %7s %10s %6s %-20s" \
      config LUT_logic CARRY FF SRL DSP BRAM36 Fmax_MHz lvls krytyczna_od]
puts [string repeat "-" 104]
foreach r $results {
    lassign $r name lut carry ff srl dsp bram fmax lvls src
    puts [format "%-16s %9d %7d %6d %5d %5d %7s %10s %6s %-20s" \
          $name $lut $carry $ff $srl $dsp $bram $fmax $lvls $src]
}
puts "OPTIM_SWEEP_END"

set csv [open $out_dir/optim_sweep.csv w]
puts $csv "config,lut_logic,carry,ff,srl,dsp,bram36,fmax_mhz,levels,critical_from"
foreach r $results { puts $csv [join $r ","] }
close $csv

puts "REPORT_DIR $out_dir"

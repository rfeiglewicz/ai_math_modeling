# =============================================================================
# sweep_pipe_targets.tcl
#
# Syntezuje wybrane rdzenie przy kilku wartosciach PIPE_TARGET, zeby bylo
# widac ile realnie kosztuje dopelnianie potoku do wspolnej latencji.
#
# UWAGA: PIPE_TARGET tylko DOPELNIA. Jesli rdzen jest glebszy niz target, to
# PAD_STAGES=0 i rdzen zostaje na swojej naturalnej glebokosci - nie da sie
# go skrocic. Takie przypadki sa w tabeli oznaczone.
#
#   vivado -mode batch -nojournal -nolog -source scripts/sweep_pipe_targets.tcl
#
# Env: CMP_PART, CMP_PERIOD, PIPE_TARGETS (domyslnie "4 8 13")
# =============================================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl        $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_compare

set part    [expr {[info exists ::env(CMP_PART)]      ? $::env(CMP_PART)      : "xc7a200tfbg484-1"}]
set period  [expr {[info exists ::env(CMP_PERIOD)]    ? $::env(CMP_PERIOD)    : 2.0}]
set targets [expr {[info exists ::env(PIPE_TARGETS)]  ? $::env(PIPE_TARGETS)  : "4 8 13"}]

file mkdir $out_dir

set xdc $out_dir/clk.xdc
set fh [open $xdc w]
puts $fh "create_clock -period $period -name clk \[get_ports clk\]"
close $fh

set common {bf16_exp2_pkg.sv bf16_pipe_pad.sv bf16_decompose.sv bf16_early_out.sv}
set src_lut    [concat $common {bf16_expe_lut_rom.sv bf16_expe_lut.sv}]
set src_hybrid [concat $common {bf16_expe_sparse_decode.sv bf16_expe_hybrid_rom.sv \
                                bf16_expe_hybrid.sv}]
set src_cut    [concat $common {bf16_expe_cut_rom.sv bf16_expe_cut.sv}]
set src_poly4  [concat $common {bf16_expe_poly4_rom.sv bf16_expe_poly4.sv}]

# {etykieta  top  zrodla  generics  naturalna_glebokosc}
# Glebokosci pochodza z scripts/pipe_depths.sh (RTL zglasza je sam przy
# elaboracji), nie sa tu przepisane z glowy.
set variants [list \
  [list "expe full-lut"    bf16_expe_lut    $src_lut    "" 4] \
  [list "expe hybrid"      bf16_expe_hybrid $src_hybrid "" 4] \
  [list "expe cut+retime"  bf16_expe_cut    $src_cut    "RETIME_CUT=2 RETIME_FE=1" 7] \
  [list "poly4+retime"     bf16_expe_poly4  $src_poly4  "RETIME_MULT=0 RETIME_FE=2" 9] \
  [list "poly4 dsp+retime" bf16_expe_poly4  $src_poly4  "DSP_FRONTEND=1 RESET_DATAPATH=0 RETIME_MULT=1 RETIME_FE=1" 12] \
]

# Wiersz z report_utilization: | Nazwa | Used | Fixed | Prohibited | Available |
proc util_row {txt name idx} {
    set pat "\\|\\s+[string map {* {\\*}} $name]\\s*\\|(\[^\n\]*)\\|"
    if {[regexp $pat $txt -> rest]} {
        set v [string trim [lindex [split $rest "|"] $idx]]
        if {[string is double -strict $v]} { return $v }
    }
    return 0
}

set results {}

foreach t $targets {
    foreach v $variants {
        lassign $v label top files generics natural

        puts "=== $label @ PIPE_TARGET=$t ==="

        close_project -quiet
        create_project -in_memory -part $part
        foreach f $files { read_verilog -sv $rtl/$f }
        read_xdc $xdc

        set rc [catch {
            synth_design -top $top -part $part \
                -generic "REGISTER_STAGES=1 PIPE_TARGET=$t $generics"
        } err]
        if {$rc} { puts "!!! $label : $err"; continue }

        set util [report_utilization -return_string]

        set lut  [util_row $util "LUT as Logic"          0]
        set srl  [util_row $util "LUT as Shift Register" 0]
        set ff   [util_row $util "Register as Flip Flop" 0]
        set dsp  [util_row $util "DSPs"                  0]
        set bram [util_row $util "Block RAM Tile"        0]
        set carry [llength [get_cells -hierarchical -quiet \
                            -filter {PRIMITIVE_GROUP == CARRY}]]

        set fmax "n/a"
        set lvls "-"
        set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
        if {[llength $tp] > 0} {
            set slack [get_property -quiet SLACK $tp]
            if {$slack ne ""} {
                set fmax [format "%.1f" [expr {1000.0/($period-$slack)}]]
                set lvls [get_property -quiet LOGIC_LEVELS $tp]
            }
        }

        # PIPE_TARGET tylko dopelnia w gore.
        set depth [expr {$t > $natural ? $t : $natural}]
        set pad   [expr {$depth - $natural}]
        set note  [expr {$t < $natural ? "za plytki target" : ""}]

        lappend results [list $t $label $depth $pad $lut $carry $ff $srl \
                              $dsp $bram $fmax $lvls $note]
    }
}

puts ""
puts "PIPE_TARGET_SWEEP_BEGIN"
puts "part   $part"
puts "period ${period} ns"
puts ""
puts [format "%-6s %-17s %6s %5s %6s %6s %6s %5s %4s %6s %9s %5s  %s" \
      target core stopnie pad LUT CARRY FF SRL DSP BRAM Fmax_MHz lvls uwaga]
puts [string repeat "-" 108]
set prev ""
foreach r $results {
    lassign $r t label depth pad lut carry ff srl dsp bram fmax lvls note
    if {$prev ne "" && $prev ne $t} { puts "" }
    set prev $t
    puts [format "%-6s %-17s %6d %5d %6d %6d %6d %5d %4d %6s %9s %5s  %s" \
          $t $label $depth $pad $lut $carry $ff $srl $dsp $bram $fmax $lvls $note]
}
puts "PIPE_TARGET_SWEEP_END"

set csv [open $out_dir/pipe_target_sweep.csv w]
puts $csv "pipe_target,core,stages,pad,lut_logic,carry,ff,srl,dsp,bram36,fmax_mhz,levels,note"
foreach r $results { puts $csv [join $r ","] }
close $csv
puts "CSV $out_dir/pipe_target_sweep.csv"

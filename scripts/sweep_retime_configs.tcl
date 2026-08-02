# =============================================================================
# sweep_retime_configs.tcl
#
# Syntezuje KAZDA sensowna konfiguracje retimingu kazdego rdzenia, zeby dalo
# sie odpowiedziec na pytanie: "przy budzecie N stopni potoku, jaka jest
# najwyzsza czestotliwosc pracy calego ukladu".
#
# Retiming to zmienna swobodna, nie stala. Rdzen z domyslnym retimingiem moze
# byc za gleboki na dany budzet, ale ten sam rdzen z mniejszym retimingiem sie
# zmiesci - wolniejszy, ale dostepny. Bez tego sweepu nie da sie tego porownac.
#
# PIPE_TARGET=0 wszedzie: dopelnianie nie zmienia Fmax (zmierzone w
# docs/pipe_target_comparison.md, dFmax=0 dla kazdego rdzenia), a tutaj
# interesuje nas naturalna glebokosc.
#
#   vivado -mode batch -nojournal -nolog -source scripts/sweep_retime_configs.tcl
#
# Env: CMP_PART, CMP_PERIOD
# =============================================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl        $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_compare

set part   [expr {[info exists ::env(CMP_PART)]   ? $::env(CMP_PART)   : "xc7a200tfbg484-1"}]
set period [expr {[info exists ::env(CMP_PERIOD)] ? $::env(CMP_PERIOD) : 2.0}]

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

# -----------------------------------------------------------------------------
# {rodzina  etykieta  top  zrodla  generics  glebokosc}
#
# Glebokosci z formul w RTL, zweryfikowane niezaleznie przez
# scripts/pipe_depths.sh (rdzen zglasza je sam przy elaboracji):
#   cut       = 4 + RETIME_CUT + RETIME_FE
#   poly4     = 3 + FE_DEPTH + 4*(1 + RETIME_MULT)
#   poly4 dsp = to samo, ale FE_EXTRA jest wymuszone na 1
#
# Dla wariantu DSP RETIME_FE=0 i =1 daja identyczna strukture (FE_EXTRA i tak
# jest 1), wiec FE=1 jest pominiete jako duplikat.
# -----------------------------------------------------------------------------
set variants [list \
  [list "full-lut" "full-lut"          bf16_expe_lut    $src_lut    "" 4] \
  [list "hybrid"   "hybrid"            bf16_expe_hybrid $src_hybrid "" 4] \
\
  [list "cut" "cut C0 F0"  bf16_expe_cut $src_cut "RETIME_CUT=0 RETIME_FE=0" 4] \
  [list "cut" "cut C1 F0"  bf16_expe_cut $src_cut "RETIME_CUT=1 RETIME_FE=0" 5] \
  [list "cut" "cut C0 F1"  bf16_expe_cut $src_cut "RETIME_CUT=0 RETIME_FE=1" 5] \
  [list "cut" "cut C2 F0"  bf16_expe_cut $src_cut "RETIME_CUT=2 RETIME_FE=0" 6] \
  [list "cut" "cut C1 F1"  bf16_expe_cut $src_cut "RETIME_CUT=1 RETIME_FE=1" 6] \
  [list "cut" "cut C2 F1"  bf16_expe_cut $src_cut "RETIME_CUT=2 RETIME_FE=1" 7] \
\
  [list "poly4" "poly4 M0 F0" bf16_expe_poly4 $src_poly4 "RETIME_MULT=0 RETIME_FE=0"  7] \
  [list "poly4" "poly4 M0 F1" bf16_expe_poly4 $src_poly4 "RETIME_MULT=0 RETIME_FE=1"  8] \
  [list "poly4" "poly4 M0 F2" bf16_expe_poly4 $src_poly4 "RETIME_MULT=0 RETIME_FE=2"  9] \
  [list "poly4" "poly4 M1 F0" bf16_expe_poly4 $src_poly4 "RETIME_MULT=1 RETIME_FE=0" 11] \
  [list "poly4" "poly4 M1 F1" bf16_expe_poly4 $src_poly4 "RETIME_MULT=1 RETIME_FE=1" 12] \
  [list "poly4" "poly4 M1 F2" bf16_expe_poly4 $src_poly4 "RETIME_MULT=1 RETIME_FE=2" 13] \
\
  [list "poly4dsp" "p4dsp M0 F0" bf16_expe_poly4 $src_poly4 "DSP_FRONTEND=1 RESET_DATAPATH=0 RETIME_MULT=0 RETIME_FE=0"  8] \
  [list "poly4dsp" "p4dsp M0 F2" bf16_expe_poly4 $src_poly4 "DSP_FRONTEND=1 RESET_DATAPATH=0 RETIME_MULT=0 RETIME_FE=2"  9] \
  [list "poly4dsp" "p4dsp M1 F0" bf16_expe_poly4 $src_poly4 "DSP_FRONTEND=1 RESET_DATAPATH=0 RETIME_MULT=1 RETIME_FE=0" 12] \
  [list "poly4dsp" "p4dsp M1 F2" bf16_expe_poly4 $src_poly4 "DSP_FRONTEND=1 RESET_DATAPATH=0 RETIME_MULT=1 RETIME_FE=2" 13] \
]

proc util_row {txt name idx} {
    set pat "\\|\\s+[string map {* {\\*}} $name]\\s*\\|(\[^\n\]*)\\|"
    if {[regexp $pat $txt -> rest]} {
        set v [string trim [lindex [split $rest "|"] $idx]]
        if {[string is double -strict $v]} { return $v }
    }
    return 0
}

set results {}

foreach v $variants {
    lassign $v family label top files generics depth

    puts "=== $label (glebokosc $depth) ==="

    close_project -quiet
    create_project -in_memory -part $part
    foreach f $files { read_verilog -sv $rtl/$f }
    read_xdc $xdc

    set rc [catch {
        synth_design -top $top -part $part \
            -generic "REGISTER_STAGES=1 PIPE_TARGET=0 $generics"
    } err]
    if {$rc} { puts "!!! $label : $err"; continue }

    set util [report_utilization -return_string]
    set lut  [util_row $util "LUT as Logic"          0]
    set srl  [util_row $util "LUT as Shift Register" 0]
    set ff   [util_row $util "Register as Flip Flop" 0]
    set dsp  [util_row $util "DSPs"                  0]
    set bram [util_row $util "Block RAM Tile"        0]

    set fmax "n/a"
    set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $tp] > 0} {
        set slack [get_property -quiet SLACK $tp]
        if {$slack ne ""} {
            set fmax [format "%.1f" [expr {1000.0/($period-$slack)}]]
        }
    }

    lappend results [list $family $label $depth $lut $ff $srl $dsp $bram $fmax $generics]
}

puts ""
puts "RETIME_SWEEP_BEGIN"
puts [format "%-10s %-14s %6s %6s %6s %5s %4s %6s %9s" \
      rodzina konfiguracja stopnie LUT FF SRL DSP BRAM Fmax_MHz]
puts [string repeat "-" 78]
foreach r $results {
    lassign $r fam label depth lut ff srl dsp bram fmax gen
    puts [format "%-10s %-14s %6d %6d %6d %5d %4d %6s %9s" \
          $fam $label $depth $lut $ff $srl $dsp $bram $fmax]
}
puts "RETIME_SWEEP_END"

set csv [open $out_dir/retime_configs.csv w]
puts $csv "family,config,stages,lut_logic,ff,srl,dsp,bram36,fmax_mhz,generics"
foreach r $results { puts $csv [join $r ","] }
close $csv
puts "CSV $out_dir/retime_configs.csv"

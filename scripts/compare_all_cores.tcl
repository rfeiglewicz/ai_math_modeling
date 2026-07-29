# =============================================================================
# compare_all_cores.tcl
#
# Syntezuje wszystkie warianty BF16 exp() obok siebie i drukuje jedna tabele.
# Kazdy rdzen dostaje wlasny in-memory project, wiec liczby nie sa skazone
# pozostalosciami po poprzednim przebiegu.
#
# Uruchomienie:
#   make compare_cores
# lub
#   vivado -mode batch -nojournal -nolog -source scripts/compare_all_cores.tcl
#
# Zmienne srodowiskowe:
#   CMP_PART   - uklad docelowy      (domyslnie xc7a200tfbg484-1)
#   CMP_PERIOD - okres zegara w ns   (domyslnie 4.0 = 250 MHz)
#   CMP_ONLY   - lista nazw rdzeni do zsyntezowania (domyslnie wszystkie)
# =============================================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]
set rtl        $repo_root/src/rtl
set out_dir    $repo_root/build/vivado_compare

set part   [expr {[info exists ::env(CMP_PART)]   ? $::env(CMP_PART)   : "xc7a200tfbg484-1"}]
# Deliberately aggressive: synthesis stops optimising once it meets the
# constraint, so a slack target the cores can actually reach would understate
# Fmax. 2 ns is unreachable for all of them, which keeps the effort maximal.
set period [expr {[info exists ::env(CMP_PERIOD)] ? $::env(CMP_PERIOD) : 2.0}]
set only   [expr {[info exists ::env(CMP_ONLY)]   ? $::env(CMP_ONLY)   : ""}]

file mkdir $out_dir

# The clock has to exist BEFORE synth_design, otherwise the whole run is
# unconstrained and timing-driven optimisation never happens.
set xdc $out_dir/clk.xdc
set fh [open $xdc w]
puts $fh "create_clock -period $period -name clk \[get_ports clk\]"
close $fh

# -----------------------------------------------------------------------------
# Wspolne zestawy plikow
# -----------------------------------------------------------------------------
set common {bf16_exp2_pkg.sv bf16_pipe_pad.sv bf16_decompose.sv bf16_early_out.sv}

set src_exp2  [concat $common {bf16_recompose.sv bf16_log2e_mult.sv \
                               bf16_unified_shift.sv bf16_coeff_rom.sv \
                               bf16_linear_approx.sv bf16_normalize.sv \
                               bf16_round.sv bf16_exp2.sv}]
set src_lut    [concat $common {bf16_expe_lut_rom.sv bf16_expe_lut.sv}]
set src_hybrid [concat $common {bf16_expe_sparse_decode.sv bf16_expe_hybrid_rom.sv \
                                bf16_expe_hybrid.sv}]
set src_cut    [concat $common {bf16_expe_cut_rom.sv bf16_expe_cut.sv}]
set src_poly4  [concat $common {bf16_expe_poly4_rom.sv bf16_expe_poly4.sv}]

# -----------------------------------------------------------------------------
# Lista wariantow:  {etykieta  top  zrodla  generics}
#
# Kazdy rdzen wystepuje dwa razy: bez retimingu i z retimingiem, zeby bylo
# widac ile kosztuje i ile daje. PIPE_TARGET=0 wszedzie, bo dopelnienie do
# wspolnej latencji nie ma wplywu na Fmax, a zaciemnia obraz zasobow.
# -----------------------------------------------------------------------------
set RT_OFF_EXP2 "RETIME_LOG2E=0 RETIME_SHIFT=0 RETIME_APPROX=0 RETIME_NORM=0 RETIME_ROUND=0 SPLIT_MULT=0"
set RT_ON_EXP2  "RETIME_LOG2E=2 RETIME_SHIFT=1 RETIME_APPROX=3 RETIME_NORM=1 RETIME_ROUND=2 SPLIT_MULT=1"
set EXP2_OPT    "DSP_SHIFT=1 MANT_MULT_ROUND_FRAC=21 RESET_DATAPATH=0"

set variants [list \
  [list "exp2 baseline"    bf16_exp2        $src_exp2   "REGISTER_STAGES=1 PIPE_TARGET=0 $RT_OFF_EXP2"] \
  [list "exp2 opt"         bf16_exp2        $src_exp2   "REGISTER_STAGES=1 PIPE_TARGET=0 $EXP2_OPT $RT_OFF_EXP2"] \
  [list "exp2 opt+retime"  bf16_exp2        $src_exp2   "REGISTER_STAGES=1 PIPE_TARGET=0 $EXP2_OPT $RT_ON_EXP2"] \
  [list "expe full-lut"    bf16_expe_lut    $src_lut    "REGISTER_STAGES=1 PIPE_TARGET=0"] \
  [list "expe hybrid"      bf16_expe_hybrid $src_hybrid "REGISTER_STAGES=1 PIPE_TARGET=0"] \
  [list "expe cut"         bf16_expe_cut    $src_cut    "REGISTER_STAGES=1 PIPE_TARGET=0 RETIME_CUT=0 RETIME_FE=0"] \
  [list "expe cut+retime"  bf16_expe_cut    $src_cut    "REGISTER_STAGES=1 PIPE_TARGET=0 RETIME_CUT=2 RETIME_FE=1"] \
  [list "expe poly4"       bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1 PIPE_TARGET=0 RETIME_MULT=0 RETIME_FE=0"] \
  [list "poly4+retime"     bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1 PIPE_TARGET=0 RETIME_MULT=1 RETIME_FE=2"] \
  [list "poly4 dsp"        bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1 PIPE_TARGET=0 DSP_FRONTEND=1 RESET_DATAPATH=0 RETIME_MULT=0 RETIME_FE=0"] \
  [list "poly4 dsp+retime" bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1 PIPE_TARGET=0 DSP_FRONTEND=1 RESET_DATAPATH=0 RETIME_MULT=1 RETIME_FE=2"] \
]

set results {}
set failed  {}

foreach v $variants {
    lassign $v label top files generics

    if {$only ne "" && [lsearch -exact $only $label] < 0} { continue }

    puts "=== syntezuje: $label ($top) ==="

    set rc [catch {
        close_project -quiet
        create_project -in_memory -part $part

        foreach f $files { read_verilog -sv $rtl/$f }
        read_xdc $xdc

        synth_design -top $top -part $part -generic $generics
    } err]

    if {$rc} {
        puts "!!! $label : $err"
        lappend failed [list $label $err]
        continue
    }

    set lut   [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == LUT}]]
    set carry [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == CARRY}]]
    set ff    [llength [get_cells -hierarchical -quiet -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    set srl   [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ SRL*}]]
    set dsp   [llength [get_cells -hierarchical -quiet -filter {REF_NAME == DSP48E1}]]
    set bram  [llength [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB*}]]

    # SRL16E sits in the LUT primitive group but is LUT-as-memory, not logic.
    # Report it separately so pipeline padding does not look like extra logic.
    set lut_logic [expr {$lut - $srl}]

    # Fmax z najgorszej sciezki wewnetrznej przy zadanym okresie.
    set fmax "n/a"
    set lvls "-"
    set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $tp] > 0} {
        set slack [get_property -quiet SLACK $tp]
        if {$slack ne ""} {
            set fmax [format "%.1f" [expr {1000.0 / ($period - $slack)}]]
            set lvls [get_property -quiet LOGIC_LEVELS $tp]
        }
    }

    set tag [string map {" " _} $label]
    report_utilization              -file $out_dir/util_${tag}.rpt
    report_utilization -hierarchical -file $out_dir/util_${tag}_hier.rpt
    report_timing -max_paths 3 -unique_pins -delay_type max \
                  -file $out_dir/crit_${tag}.rpt

    lappend results [list $label $lut_logic $carry $ff $srl $dsp $bram $fmax $lvls]
}

# -----------------------------------------------------------------------------
# Tabela zbiorcza
# -----------------------------------------------------------------------------
puts ""
puts "CORE_COMPARISON_BEGIN"
puts "part   $part"
puts "period ${period} ns"
puts ""
puts [format "%-17s %9s %7s %7s %6s %5s %6s %10s %6s" \
      core LUT_logic CARRY FF SRL DSP BRAM Fmax_MHz lvls]
puts [string repeat "-" 82]
foreach r $results {
    lassign $r label lut carry ff srl dsp bram fmax lvls
    puts [format "%-17s %9d %7d %7d %6d %5d %6d %10s %6s" \
          $label $lut $carry $ff $srl $dsp $bram $fmax $lvls]
}
if {[llength $failed] > 0} {
    puts ""
    puts "NIEUDANE:"
    foreach f $failed { puts "  [lindex $f 0]" }
}
puts "CORE_COMPARISON_END"
puts "REPORT_DIR $out_dir"

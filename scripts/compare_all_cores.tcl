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
set period [expr {[info exists ::env(CMP_PERIOD)] ? $::env(CMP_PERIOD) : 4.0}]
set only   [expr {[info exists ::env(CMP_ONLY)]   ? $::env(CMP_ONLY)   : ""}]

file mkdir $out_dir

# -----------------------------------------------------------------------------
# Wspolne zestawy plikow
# -----------------------------------------------------------------------------
set common {bf16_exp2_pkg.sv bf16_decompose.sv bf16_early_out.sv}

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
# -----------------------------------------------------------------------------
set variants [list \
  [list "exp2 baseline"   bf16_exp2        $src_exp2   "REGISTER_STAGES=1"] \
  [list "exp2 dsp-shift"  bf16_exp2        $src_exp2   "REGISTER_STAGES=1 DSP_SHIFT=1"] \
  [list "exp2 optimised"  bf16_exp2        $src_exp2   "REGISTER_STAGES=1 DSP_SHIFT=1 MANT_MULT_ROUND_FRAC=21 RESET_DATAPATH=0"] \
  [list "expe full-lut"   bf16_expe_lut    $src_lut    "REGISTER_STAGES=1"] \
  [list "expe hybrid"     bf16_expe_hybrid $src_hybrid "REGISTER_STAGES=1"] \
  [list "expe cut-ladder" bf16_expe_cut    $src_cut    "REGISTER_STAGES=1"] \
  [list "expe poly4"      bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1"] \
  [list "expe poly4 dsp"  bf16_expe_poly4  $src_poly4  "REGISTER_STAGES=1 DSP_FRONTEND=1 RESET_DATAPATH=0"] \
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

        synth_design -top $top -part $part -generic $generics
        create_clock -period $period -name clk [get_ports clk]
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

    # Fmax z najgorszej sciezki wewnetrznej przy zadanym okresie.
    set fmax "n/a"
    set tp [get_timing_paths -quiet -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $tp] > 0} {
        set slack [get_property -quiet SLACK $tp]
        if {$slack ne "" && $slack < $period} {
            set fmax [format "%.1f" [expr {1000.0 / ($period - $slack)}]]
        }
    }

    set tag [string map {" " _} $label]
    report_utilization              -file $out_dir/util_${tag}.rpt
    report_utilization -hierarchical -file $out_dir/util_${tag}_hier.rpt

    lappend results [list $label $lut $carry $ff $srl $dsp $bram $fmax]
}

# -----------------------------------------------------------------------------
# Tabela zbiorcza
# -----------------------------------------------------------------------------
puts ""
puts "CORE_COMPARISON_BEGIN"
puts "part   $part"
puts "period ${period} ns"
puts ""
puts [format "%-16s %7s %7s %7s %6s %5s %6s %10s" \
      core LUT CARRY FF SRL DSP BRAM Fmax_MHz]
puts [string repeat "-" 72]
foreach r $results {
    lassign $r label lut carry ff srl dsp bram fmax
    puts [format "%-16s %7d %7d %7d %6d %5d %6d %10s" \
          $label $lut $carry $ff $srl $dsp $bram $fmax]
}
if {[llength $failed] > 0} {
    puts ""
    puts "NIEUDANE:"
    foreach f $failed { puts "  [lindex $f 0]" }
}
puts "CORE_COMPARISON_END"
puts "REPORT_DIR $out_dir"

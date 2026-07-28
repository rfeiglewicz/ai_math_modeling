# Batch synthesis of the DSP-heavy BF16 exp(x) implementation.
# Can be run from any working directory: every path below is resolved relative
# to this script, not to the shell's cwd.

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]

set part xc7a200tfbg484-1
set top  bf16_expe_poly4
set out_dir [file join $repo_root build vivado_poly4_dsp]
set rtl_dir [file join $repo_root src rtl]

file mkdir $out_dir

read_verilog -sv [list \
    [file join $rtl_dir bf16_exp2_pkg.sv] \
    [file join $rtl_dir bf16_decompose.sv] \
    [file join $rtl_dir bf16_early_out.sv] \
    [file join $rtl_dir bf16_expe_poly4_rom.sv] \
    [file join $rtl_dir bf16_expe_poly4.sv]]

synth_design \
    -top $top \
    -part $part \
    -generic {REGISTER_STAGES=1 DSP_FRONTEND=1}

write_checkpoint -force [file join $out_dir bf16_expe_poly4_dsp_synth.dcp]
report_utilization \
    -hierarchical \
    -file [file join $out_dir bf16_expe_poly4_dsp_utilization_hier.rpt]
report_utilization \
    -file [file join $out_dir bf16_expe_poly4_dsp_utilization.rpt]
report_timing_summary \
    -file [file join $out_dir bf16_expe_poly4_dsp_timing_synth.rpt]

set dsp_cells [get_cells -hierarchical -filter {REF_NAME == DSP48E1}]
puts "SYNTHESIS_COMPLETE"
puts "REPORT_DIR=$out_dir"
puts "DSP48E1_COUNT=[llength $dsp_cells]"
puts "DSP48E1_CELLS=$dsp_cells"

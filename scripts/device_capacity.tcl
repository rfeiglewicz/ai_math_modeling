# =============================================================================
# device_capacity.tcl
#
# Asks Vivado how much of each primitive the target part actually has, rather
# than quoting a datasheet. Counts sites and multiplies by the per-site
# capacity, which is how report_utilization derives its own totals.
#
#   vivado -mode batch -nojournal -nolog -source scripts/device_capacity.tcl
#
# Env: CMP_PART - target device (default xc7a200tfbg484-1)
# =============================================================================

set part [expr {[info exists ::env(CMP_PART)] ? $::env(CMP_PART) : "xc7a200tfbg484-1"}]

close_project -quiet
create_project -in_memory -part $part

set slicel [llength [get_sites -quiet -filter {SITE_TYPE == SLICEL}]]
set slicem [llength [get_sites -quiet -filter {SITE_TYPE == SLICEM}]]
set slices [expr {$slicel + $slicem}]

# 7-series slice: 4 LUT6, 8 flip-flops, 1 CARRY4 chain.
# Only SLICEM LUTs can be used as SRL.
set luts   [expr {$slices * 4}]
set ffs    [expr {$slices * 8}]
set carry  $slices
set srls   [expr {$slicem * 4}]

set dsp    [llength [get_sites -quiet -filter {SITE_TYPE =~ DSP48*}]]
set rb18   [llength [get_sites -quiet -filter {SITE_TYPE =~ RAMBFIFO18* || SITE_TYPE =~ RAMB18*}]]
set rb36   [llength [get_sites -quiet -filter {SITE_TYPE =~ RAMBFIFO36* || SITE_TYPE =~ RAMB36*}]]

puts ""
puts "DEVICE_CAPACITY_BEGIN"
puts "part        $part"
puts "SLICE       $slices  (SLICEL $slicel + SLICEM $slicem)"
puts "LUT         $luts"
puts "FF          $ffs"
puts "CARRY4      $carry"
puts "SRL_capable $srls"
puts "DSP48E1     $dsp"
puts "RAMB18E1    $rb18"
puts "RAMB36E1    $rb36"
puts "DEVICE_CAPACITY_END"

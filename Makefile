# Compiler settings
CXX = g++
# Zakładam, że ac_datatypes jest w libs/ac_types. Dostosuj tę ścieżkę w razie potrzeby.
AC_TYPES_DIR = ac_types
CXXFLAGS = -std=c++17 -Wall -Wextra -I./src/utils -I./src/approximations -I$(AC_TYPES_DIR)/include

# Directories
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

# Target executable names
TARGET_MAIN = $(BUILD_DIR)/fp_utils_test
TARGET_EXHAUSTIVE = $(BUILD_DIR)/fp_utils_exhaustive_test
TARGET_GEN_APPROX = $(BUILD_DIR)/gen_bf16_exp2_approx
TARGET_ULP_ANALYSIS = $(BUILD_DIR)/ulp_error_analysis
TARGET_LINEAR_APPROX = $(BUILD_DIR)/test_bf16_linear_approx
TARGET_GEN_PACKED = $(BUILD_DIR)/gen_packed_coeffs
TARGET_RNE_SWEEP = $(BUILD_DIR)/rne_rounding_sweep
TARGET_PWL_OPTIM = $(BUILD_DIR)/exp_pwl_optim
TARGET_GEN_OPTIM_COEFFS = $(BUILD_DIR)/gen_optim_coeffs
TARGET_EXP2_OPTIM_TEST = $(BUILD_DIR)/bf16_exp2_optim_test
TARGET_GEN_EXPE_LUT = $(BUILD_DIR)/gen_bf16_expe_lut
TARGET_EXPE_LUT_TEST = $(BUILD_DIR)/bf16_expe_lut_test
TARGET_GEN_EXPE_LUT_APPROX = $(BUILD_DIR)/gen_bf16_expe_lut_approx

# Source files
TEST_SRC_MAIN = $(TEST_DIR)/fp_utils_test.cpp
TEST_SRC_EXHAUSTIVE = $(TEST_DIR)/fp_utils_exhaustive_test.cpp
TEST_SRC_GEN_APPROX = $(TEST_DIR)/gen_bf16_exp2_approx.cpp
TEST_SRC_ULP_ANALYSIS = $(TEST_DIR)/ulp_error_analysis.cpp
TEST_SRC_LINEAR_APPROX = $(TEST_DIR)/test_bf16_linear_approx.cpp
SRC_GEN_PACKED = modeling/coeff_gen/gen_packed_coeffs.cpp
TEST_SRC_RNE_SWEEP = $(TEST_DIR)/rne_rounding_sweep.cpp
TEST_SRC_PWL_OPTIM = $(TEST_DIR)/exp_pwl_optim.cpp
SRC_GEN_OPTIM_COEFFS = modeling/coeff_gen/gen_optim_coeffs.cpp
TEST_SRC_EXP2_OPTIM = $(TEST_DIR)/bf16_exp2_optim_test.cpp
TEST_SRC_GEN_EXPE_LUT = $(TEST_DIR)/gen_bf16_expe_lut.cpp
TEST_SRC_EXPE_LUT_TEST = $(TEST_DIR)/bf16_expe_lut_test.cpp
TEST_SRC_GEN_EXPE_LUT_APPROX = $(TEST_DIR)/gen_bf16_expe_lut_approx.cpp

# Default rule: build all
.PHONY: all run run_exhaustive gen_approx ulp_analysis run_linear_approx gen_packed rne_sweep exp_pwl_optim gen_optim_coeffs exp2_optim clean clean_rtl clean_vivado clean_all

all: $(TARGET_MAIN) $(TARGET_EXHAUSTIVE) $(TARGET_GEN_APPROX) $(TARGET_ULP_ANALYSIS) $(TARGET_LINEAR_APPROX) $(TARGET_GEN_PACKED) $(TARGET_RNE_SWEEP)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build rules
$(TARGET_MAIN): $(TEST_SRC_MAIN) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_EXHAUSTIVE): $(TEST_SRC_EXHAUSTIVE) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_GEN_APPROX): $(TEST_SRC_GEN_APPROX) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_ULP_ANALYSIS): $(TEST_SRC_ULP_ANALYSIS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_LINEAR_APPROX): $(TEST_SRC_LINEAR_APPROX) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_GEN_PACKED): $(SRC_GEN_PACKED) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Imodeling/coeff_gen -o $@ $<

$(TARGET_RNE_SWEEP): $(TEST_SRC_RNE_SWEEP) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 -o $@ $<

$(TARGET_PWL_OPTIM): $(TEST_SRC_PWL_OPTIM) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 -o $@ $<

$(TARGET_GEN_OPTIM_COEFFS): $(SRC_GEN_OPTIM_COEFFS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Imodeling/coeff_gen -o $@ $<

$(TARGET_EXP2_OPTIM_TEST): $(TEST_SRC_EXP2_OPTIM) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 -o $@ $<

$(TARGET_GEN_EXPE_LUT): $(TEST_SRC_GEN_EXPE_LUT) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_EXPE_LUT_TEST): $(TEST_SRC_EXPE_LUT_TEST) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 -o $@ $<

$(TARGET_GEN_EXPE_LUT_APPROX): $(TEST_SRC_GEN_EXPE_LUT_APPROX) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Run rules
run: $(TARGET_MAIN)
	./$(TARGET_MAIN)

run_exhaustive: $(TARGET_EXHAUSTIVE)
	./$(TARGET_EXHAUSTIVE)

gen_approx: $(TARGET_GEN_APPROX)
	./$(TARGET_GEN_APPROX)

ulp_analysis: $(TARGET_ULP_ANALYSIS)
	./$(TARGET_ULP_ANALYSIS)

# Generate the tabulated (full-LUT) expe output for the ULP analyzer.
gen_expe_lut_approx: $(TARGET_GEN_EXPE_LUT_APPROX) gen_expe_lut
	./$(TARGET_GEN_EXPE_LUT_APPROX)

gen_packed: $(TARGET_GEN_PACKED)
	./$(TARGET_GEN_PACKED)

rne_sweep: $(TARGET_RNE_SWEEP)
	./$(TARGET_RNE_SWEEP)

# Datapath width optimisation study for the exp2/expe PWL core.
# Finds the narrowest log2(e) product, unified shift, polynomial input and
# b - a*x accumulator that stay bit-exact with bf16_exp2_approx<29>.
exp_pwl_optim: $(TARGET_PWL_OPTIM)
	./$(TARGET_PWL_OPTIM)

# Regenerate the narrow coefficient ROM (a Q0.17, b Q0.18) from the shipped
# Q1.20 table. Writes modeling/coeff_gen/bf16_exp2_optim_coeffs.hpp.
gen_optim_coeffs: $(TARGET_GEN_OPTIM_COEFFS)
	./$(TARGET_GEN_OPTIM_COEFFS)

# Width-optimised standalone exp2/expe core: exhaustive equivalence check
# against the production model. Independent of bf16_exp2.hpp / bf16_exp2_core.hpp.
exp2_optim: $(TARGET_EXP2_OPTIM_TEST)
	./$(TARGET_EXP2_OPTIM_TEST)

# Generate the full expe LUT header, then build & run the ULP verification test.
gen_expe_lut: $(TARGET_GEN_EXPE_LUT)
	./$(TARGET_GEN_EXPE_LUT)

expe_lut_test: gen_expe_lut $(TARGET_EXPE_LUT_TEST)
	./$(TARGET_EXPE_LUT_TEST)

run_linear_approx: $(TARGET_LINEAR_APPROX)
	./$(TARGET_LINEAR_APPROX)

# -------------------------------------------------------------------------
# Cleaning.
#
# `clean` removes the C++ build output only. `clean_rtl` removes the Verilator
# obj_dirs -- there is one per configuration (obj_dir_optim_hu, obj_dir_opt_pl,
# obj_dir_poly4_dsp_axi, ...) at roughly 700 KB each, so they add up fast.
# `clean_vivado` removes the synthesis runs and the scratch state Vivado leaves
# behind, including in scripts/ when it is launched from there.
#
# Use these instead of `git clean -fd`: new sources are untracked until their
# first commit, and a blind clean deletes them along with the artefacts.
# -------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)

clean_rtl:
	rm -rf obj_dir obj_dir_*

clean_vivado:
	rm -rf .Xil scripts/.Xil scripts/build
	rm -f vivado*.jou vivado*.log scripts/vivado*.jou scripts/vivado*.log

clean_all: clean clean_rtl clean_vivado

# =========================================================================
# RTL Verification (SystemVerilog + Verilator)
# =========================================================================
RTL_DIR    = src/rtl
RTL_TB     = tests/rtl_compliance_test.cpp
VERILATOR  = verilator
VER_FLAGS  = -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
             --top-module bf16_exp2 --cc
VER_INC    = -I$(RTL_DIR)
VER_CFLAGS = -CFLAGS "-I../src/utils -I../src/approximations \
             -I../$(AC_TYPES_DIR)/include -I../modeling/coeff_gen"
RTL_SRC    = $(RTL_DIR)/bf16_exp2_pkg.sv \
               $(RTL_DIR)/bf16_pipe_pad.sv \
             $(RTL_DIR)/bf16_decompose.sv \
             $(RTL_DIR)/bf16_recompose.sv \
             $(RTL_DIR)/bf16_early_out.sv \
             $(RTL_DIR)/bf16_log2e_mult.sv \
             $(RTL_DIR)/bf16_unified_shift.sv \
             $(RTL_DIR)/bf16_coeff_rom.sv \
             $(RTL_DIR)/bf16_linear_approx.sv \
             $(RTL_DIR)/bf16_normalize.sv \
             $(RTL_DIR)/bf16_round.sv \
             $(RTL_DIR)/bf16_exp2.sv

.PHONY: rtl_verify rtl_verify_pipelined verilate rtl_axi_test

verilate:
	$(VERILATOR) $(VER_FLAGS) $(VER_INC) $(RTL_SRC) --exe $(RTL_TB) $(VER_CFLAGS)
	make -j -C obj_dir -f Vbf16_exp2.mk Vbf16_exp2

rtl_verify: verilate
	./obj_dir/Vbf16_exp2

rtl_verify_pipelined:
	@echo "Building pipelined RTL (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_pl $(VER_INC) $(RTL_SRC) \
	    --exe $(RTL_TB) $(VER_CFLAGS)
	make -j -C obj_dir_pl -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_pl/Vbf16_exp2 --latency 16

rtl_axi_test:
	@echo "Building AXI-Stream protocol test (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_axi $(VER_INC) $(RTL_SRC) \
	    --exe tests/rtl_axi_stream_test.cpp $(VER_CFLAGS)
	make -j -C obj_dir_axi -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_axi/Vbf16_exp2

# =========================================================================
# Width-optimised core: bf16_exp2_optim
#
# Independent RTL for the narrow datapath found by tests/exp_pwl_optim.cpp:
# log2(e) product 23 b, unified shift 30 b, x 17 b, a 17 b, b 18 b,
# a*x 34 b, accumulator 18 b, normalised mantissa 19 b, ROM 128 x 35 b.
# No priority encoder and no barrel shifter in the normaliser.
#
# Only the datapath is new. bf16_decompose, bf16_early_out, bf16_recompose
# and bf16_pipe_pad are BF16 format plumbing, identical in both cores, and
# are instantiated unmodified -- as is bf16_exp2_pkg, for fp_raw_t and the
# canonical early-out patterns.
#
# The testbench compares against bf16_exp2_approx(), the PRODUCTION model,
# because the claim being tested is that nothing changed.
# =========================================================================
RTL_OPTIM_TB  = tests/rtl_optim_compliance_test.cpp
VER_OPTIM_FLAGS = -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
                  --top-module bf16_exp2_optim --cc
RTL_OPTIM_SRC = $(RTL_DIR)/bf16_exp2_pkg.sv \
                $(RTL_DIR)/bf16_exp2_optim_pkg.sv \
                $(RTL_DIR)/bf16_pipe_pad.sv \
                $(RTL_DIR)/bf16_decompose.sv \
                $(RTL_DIR)/bf16_recompose.sv \
                $(RTL_DIR)/bf16_early_out.sv \
                $(RTL_DIR)/bf16_exp2_optim_rom.sv \
                $(RTL_DIR)/bf16_exp2_optim_log2e_mult.sv \
                $(RTL_DIR)/bf16_exp2_optim_shift.sv \
                $(RTL_DIR)/bf16_exp2_optim_approx.sv \
                $(RTL_DIR)/bf16_exp2_optim_normalize.sv \
                $(RTL_DIR)/bf16_exp2_optim_round.sv \
                $(RTL_DIR)/bf16_exp2_optim.sv

.PHONY: rtl_optim_verify rtl_optim_verify_pipelined rtl_optim_static_norm \
        rtl_optim_half_up rtl_optim_no_reset rtl_optim_round_frac_sweep rtl_optim_all

rtl_optim_verify:
	@echo "Building combinational optimised RTL (REGISTER_STAGES=0) ..."
	$(VERILATOR) $(VER_OPTIM_FLAGS) -Mdir obj_dir_optim $(VER_INC) $(RTL_OPTIM_SRC) \
	    --exe $(RTL_OPTIM_TB) $(VER_CFLAGS)
	make -j -C obj_dir_optim -f Vbf16_exp2_optim.mk Vbf16_exp2_optim
	./obj_dir_optim/Vbf16_exp2_optim

rtl_optim_verify_pipelined:
	@echo "Building pipelined optimised RTL (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(VER_OPTIM_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_optim_pl \
	    $(VER_INC) $(RTL_OPTIM_SRC) --exe $(RTL_OPTIM_TB) $(VER_CFLAGS)
	make -j -C obj_dir_optim_pl -f Vbf16_exp2_optim.mk Vbf16_exp2_optim
	./obj_dir_optim_pl/Vbf16_exp2_optim --latency 16

# The static normaliser drops the guard bit test and shifts by a constant 1.
# It is only correct because b - a*x >= 0.5 for every reachable input, which
# is an input-space property rather than an identity -- so it gets its own
# exhaustive run instead of being trusted.
rtl_optim_static_norm:
	@echo "Building pipelined optimised RTL (STATIC_NORM=1) ..."
	$(VERILATOR) $(VER_OPTIM_FLAGS) -GREGISTER_STAGES=1 -GSTATIC_NORM=1 \
	    -Mdir obj_dir_optim_sn $(VER_INC) $(RTL_OPTIM_SRC) --exe $(RTL_OPTIM_TB) $(VER_CFLAGS)
	make -j -C obj_dir_optim_sn -f Vbf16_exp2_optim.mk Vbf16_exp2_optim
	./obj_dir_optim_sn/Vbf16_exp2_optim --latency 16

# Round-half-up instead of RNE on the log2(e) product. Reaches the same 21-bit
# minimum, and with the retiming maxed out it moves the critical path off the
# multiply: 164.8 -> 177.5 MHz on xc7a200t for one extra LUT (make sweep_optim).
# It is a different rounding rule from the C++ model, so it gets a full
# exhaustive run of its own rather than riding on the RNE result.
rtl_optim_half_up:
	@echo "Building pipelined optimised RTL (ROUND_MODE=1, half-up) ..."
	$(VERILATOR) $(VER_OPTIM_FLAGS) -GREGISTER_STAGES=1 -GROUND_MODE=1 \
	    -Mdir obj_dir_optim_hu $(VER_INC) $(RTL_OPTIM_SRC) --exe $(RTL_OPTIM_TB) $(VER_CFLAGS)
	make -j -C obj_dir_optim_hu -f Vbf16_exp2_optim.mk Vbf16_exp2_optim
	./obj_dir_optim_hu/Vbf16_exp2_optim --latency 16

# No reset on the datapath registers, so Vivado can absorb them into DSP48
# and BRAM. Must stay bit-exact: stale values are always flagged invalid.
# Measured worth: 164.8 vs 139.7 MHz and 272 vs 373 FF (make sweep_optim).
rtl_optim_no_reset:
	@echo "Building pipelined optimised RTL (RESET_DATAPATH=0) ..."
	$(VERILATOR) $(VER_OPTIM_FLAGS) -GREGISTER_STAGES=1 -GRESET_DATAPATH=0 \
	    -Mdir obj_dir_optim_nr $(VER_INC) $(RTL_OPTIM_SRC) --exe $(RTL_OPTIM_TB) $(VER_CFLAGS)
	make -j -C obj_dir_optim_nr -f Vbf16_exp2_optim.mk Vbf16_exp2_optim
	./obj_dir_optim_nr/Vbf16_exp2_optim --latency 16

# Re-derive the minimum log2(e) product width from the RTL itself, rather than
# trusting the C++ study. Measured (both modes, all 65536 patterns, compared
# against bf16_exp2_approx):
#
#   ROUND_MODE=0 RNE       21 PASS   20 FAIL   19 PASS
#   ROUND_MODE=1 half-up   21 PASS   20 FAIL
#   ROUND_MODE=2 truncate  21 FAIL
#
# 19 passing is not a licence to use it. The sweep is not monotone, because
# the re-quantisation error has a sign and can cancel against the table's own
# error at some widths and not others. 21 is the smallest width at and ABOVE
# which every width passes, which is the only property worth building on.
#
# Truncation at 21 fails, as the study predicted -- it needs 22, and 22 is not
# reachable here: MANT_MULT_F is this module's own output field width, so
# keeping more fractional bits would mean widening the whole downstream
# datapath. That is why the core rounds instead of truncating.
OPTIM_FRAC_LIST ?= 21 20 19
OPTIM_ROUND_MODE ?= 0

rtl_optim_round_frac_sweep:
	@for F in $(OPTIM_FRAC_LIST); do \
	    $(VERILATOR) $(VER_OPTIM_FLAGS) -GMANT_MULT_ROUND_FRAC=$$F \
	        -GROUND_MODE=$(OPTIM_ROUND_MODE) -Mdir obj_dir_optim_rf $(VER_INC) \
	        $(RTL_OPTIM_SRC) --exe $(RTL_OPTIM_TB) $(VER_CFLAGS) > /dev/null 2>&1 ; \
	    make -j -s -C obj_dir_optim_rf -f Vbf16_exp2_optim.mk Vbf16_exp2_optim > /dev/null 2>&1 ; \
	    printf 'MANT_MULT_ROUND_FRAC=%-3s ROUND_MODE=%s ' $$F $(OPTIM_ROUND_MODE) ; \
	    ./obj_dir_optim_rf/Vbf16_exp2_optim 2>&1 | grep -E '^\[sweep|OVERALL' | tr '\n' ' ' ; \
	    echo ; \
	    rm -rf obj_dir_optim_rf ; \
	done

rtl_optim_all: rtl_optim_verify rtl_optim_verify_pipelined rtl_optim_static_norm \
               rtl_optim_half_up rtl_optim_no_reset

# -------------------------------------------------------------------------
# Same design with the fabric barrel shifter replaced by a one-hot DSP
# multiply (bf16_unified_shift DSP_SHIFT=1). Must be bit-exact.
# -------------------------------------------------------------------------
.PHONY: rtl_dsp_shift_verify rtl_dsp_shift_verify_pipelined \
        rtl_dsp_shift_axi_test rtl_dsp_shift_all

rtl_dsp_shift_verify:
	@echo "Building combinational RTL (DSP_SHIFT=1) ..."
	$(VERILATOR) $(VER_FLAGS) -GDSP_SHIFT=1 -Mdir obj_dir_dsps $(VER_INC) $(RTL_SRC) \
	    --exe $(RTL_TB) $(VER_CFLAGS)
	make -j -C obj_dir_dsps -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_dsps/Vbf16_exp2

rtl_dsp_shift_verify_pipelined:
	@echo "Building pipelined RTL (REGISTER_STAGES=1 DSP_SHIFT=1) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 -GDSP_SHIFT=1 -Mdir obj_dir_dsps_pl \
	    $(VER_INC) $(RTL_SRC) --exe $(RTL_TB) $(VER_CFLAGS)
	make -j -C obj_dir_dsps_pl -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_dsps_pl/Vbf16_exp2 --latency 16

rtl_dsp_shift_axi_test:
	@echo "Building AXI-Stream protocol test (REGISTER_STAGES=1 DSP_SHIFT=1) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 -GDSP_SHIFT=1 -Mdir obj_dir_dsps_axi \
	    $(VER_INC) $(RTL_SRC) --exe tests/rtl_axi_stream_test.cpp $(VER_CFLAGS)
	make -j -C obj_dir_dsps_axi -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_dsps_axi/Vbf16_exp2

rtl_dsp_shift_all: rtl_dsp_shift_verify rtl_dsp_shift_verify_pipelined rtl_dsp_shift_axi_test

# -------------------------------------------------------------------------
# Sweep MANT_MULT_ROUND_FRAC through the RTL (not just the C++ model) to find
# the narrowest log2(e) product that is still bit-exact against the golden
# model. Usage:  make rtl_round_frac_sweep [FRAC_LIST="23 22 21 20"]
# -------------------------------------------------------------------------
FRAC_LIST ?= 29 25 23 22 21 20 19
.PHONY: rtl_round_frac_sweep

rtl_round_frac_sweep:
	@for F in $(FRAC_LIST); do \
	    $(VERILATOR) $(VER_FLAGS) -GDSP_SHIFT=1 -GMANT_MULT_ROUND_FRAC=$$F \
	        -Mdir obj_dir_rf $(VER_INC) $(RTL_SRC) --exe $(RTL_TB) $(VER_CFLAGS) \
	        > /dev/null 2>&1 ; \
	    make -j -s -C obj_dir_rf -f Vbf16_exp2.mk Vbf16_exp2 > /dev/null 2>&1 ; \
	    printf 'MANT_MULT_ROUND_FRAC=%-3s ' $$F ; \
	    ./obj_dir_rf/Vbf16_exp2 2>&1 | grep -E '^\[|OVERALL' | tr '\n' ' ' ; \
	    echo ; \
	    rm -rf obj_dir_rf ; \
	done

# -------------------------------------------------------------------------
# Fully optimised bf16_exp2: DSP one-hot shifter + narrowed log2(e) product
# + no reset on datapath registers. Must stay bit-exact.
# -------------------------------------------------------------------------
OPT_GENERICS = -GDSP_SHIFT=1 -GMANT_MULT_ROUND_FRAC=21 -GRESET_DATAPATH=0
.PHONY: rtl_opt_verify rtl_opt_verify_pipelined rtl_opt_axi_test rtl_opt_all

rtl_opt_verify:
	@echo "Building combinational RTL (optimised) ..."
	$(VERILATOR) $(VER_FLAGS) $(OPT_GENERICS) -Mdir obj_dir_opt $(VER_INC) $(RTL_SRC) \
	    --exe $(RTL_TB) $(VER_CFLAGS)
	make -j -C obj_dir_opt -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_opt/Vbf16_exp2

rtl_opt_verify_pipelined:
	@echo "Building pipelined RTL (optimised) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 $(OPT_GENERICS) -Mdir obj_dir_opt_pl \
	    $(VER_INC) $(RTL_SRC) --exe $(RTL_TB) $(VER_CFLAGS)
	make -j -C obj_dir_opt_pl -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_opt_pl/Vbf16_exp2 --latency 16

rtl_opt_axi_test:
	@echo "Building AXI-Stream protocol test (optimised) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 $(OPT_GENERICS) -Mdir obj_dir_opt_axi \
	    $(VER_INC) $(RTL_SRC) --exe tests/rtl_axi_stream_test.cpp $(VER_CFLAGS)
	make -j -C obj_dir_opt_axi -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_opt_axi/Vbf16_exp2

rtl_opt_all: rtl_opt_verify rtl_opt_verify_pipelined rtl_opt_axi_test

.PHONY: sweep_exp2
sweep_exp2:
	$(VIVADO) -mode batch -nojournal -nolog -source scripts/sweep_exp2_configs.tcl

# Przemiatanie konfiguracji rdzenia o zwezonej sciezce danych: tryb
# zaokraglania iloczynu log2(e), strzezona vs statyczna normalizacja, reset
# sciezki danych. Kazda z tych opcji jest w RTL opisana jako teza -- ten cel
# ja mierzy.
.PHONY: sweep_optim
sweep_optim:
	$(VIVADO) -mode batch -nojournal -nolog -source scripts/sweep_optim_configs.tcl

# =========================================================================
# Porownanie wszystkich implementacji obok siebie
# =========================================================================
.PHONY: compare_cores compare_cores_verify compare_all pipe_depths resource_table \
        sweep_pipe_targets pipe_target_table

# Synteza wszystkich rdzeni w jednym przebiegu -> jedna tabela zasobow.
compare_cores: gen_expe_lut_rom gen_expe_hybrid_tables gen_expe_cut_tables gen_expe_poly4_tables
	$(VIVADO) -mode batch -nojournal -nolog -source scripts/compare_all_cores.tcl

# Glebokosc potoku kazdego wariantu, zgloszona przez sam RTL przy elaboracji.
pipe_depths:
	@scripts/pipe_depths.sh

# docs/resource_comparison.md z pomiarow: zasoby+Fmax z Vivado, glebokosc z RTL.
# Wymaga wczesniejszego `make compare_cores` (zapisuje resources.csv).
resource_table:
	@python3 scripts/make_resource_table.py

# Ten sam rdzen przy roznych PIPE_TARGET -> ile kosztuje dopelnianie potoku.
# PIPE_TARGETS="4 8 13" zeby zmienic zestaw.
sweep_pipe_targets: gen_expe_lut_rom gen_expe_hybrid_tables gen_expe_cut_tables gen_expe_poly4_tables
	$(VIVADO) -mode batch -nojournal -nolog -source scripts/sweep_pipe_targets.tcl

pipe_target_table:
	@python3 scripts/make_pipe_target_table.py

# docs/throughput_analysis.md: ile rdzeni zmiesci sie w ukladzie i jaka daja
# laczna przepustowosc. Czyta resources.csv + pipe_target_sweep.csv, wiec
# wymaga wczesniejszego `make compare_cores` i `make sweep_pipe_targets`.
#
# Solwer calkowitoliczbowy potrzebuje scipy (HiGHS), a systemowy python3 zwykle
# go nie ma. Szukamy wiec interpretera, ktory scipy widzi: najpierw $(PYTHON),
# potem aktywne srodowisko conda, na koncu python3/python z PATH. Mozna
# wymusic recznie: make throughput_table PYTHON=/sciezka/do/python
PYTHON ?= python3

throughput_table:
	@py=""; \
	for p in "$(PYTHON)" "$(CONDA_PREFIX)/bin/python" python3 python; do \
	    [ -n "$$p" ] || continue; \
	    if "$$p" -c 'import scipy.optimize' >/dev/null 2>&1; then py="$$p"; break; fi; \
	done; \
	if [ -z "$$py" ]; then \
	    echo "make throughput_table: nie znalazlem interpretera ze scipy." >&2; \
	    echo "  aktywuj srodowisko (np. conda activate py312) albo podaj:" >&2; \
	    echo "  make throughput_table PYTHON=/sciezka/do/python" >&2; \
	    exit 1; \
	fi; \
	"$$py" scripts/make_throughput_table.py

# ---------------------------------------------------------------------------
# Punkt odniesienia na GPU: zmierzona przepustowosc exp() w BF16 na krzemie.
# Sluzy jako kontrapunkt do liczb FPGA, ktore pochodza tylko z syntezy.
#
# nvcc czesto nie jest w PATH, wiec sprawdzamy tez domyslna sciezke instalacji.
# -arch=native kompiluje pod faktycznie zainstalowane GPU; hexp/h2exp w BF16
# wymagaja sm_80 lub nowszego.
NVCC ?= nvcc

.PHONY: cuda_throughput cuda_throughput_run

cuda_throughput: build/bf16_exp_throughput

build/bf16_exp_throughput: cuda/bf16_exp_throughput.cu
	@nv=""; \
	for c in "$(NVCC)" /usr/local/cuda/bin/nvcc; do \
	    [ -n "$$c" ] || continue; \
	    if command -v "$$c" >/dev/null 2>&1; then nv="$$c"; break; fi; \
	done; \
	if [ -z "$$nv" ]; then \
	    echo "make cuda_throughput: nie znalazlem nvcc." >&2; \
	    echo "  podaj recznie: make cuda_throughput NVCC=/sciezka/do/nvcc" >&2; \
	    exit 1; \
	fi; \
	mkdir -p build; \
	echo "$$nv -O3 -std=c++17 -arch=native -o $@ $<"; \
	"$$nv" -O3 -std=c++17 -arch=native -o $@ $<

# Pelny przebieg: pomiar + przemiatanie N + wyczerpujace sprawdzenie ULP.
# CUDA_ARGS pozwala dolozyc opcje, np. make cuda_throughput_run CUDA_ARGS=--samples=256M
cuda_throughput_run: build/bf16_exp_throughput
	./build/bf16_exp_throughput --sweep $(CUDA_ARGS)

# Weryfikacja funkcjonalna wszystkich rdzeni (Verilator, wyczerpujaca).
# Rdzenie maja rozne testbenche, wiec podsumowanie filtrujemy po obu formatach:
# "checked=/OVERALL" (exp2, lut) oraz "Checked:/Mismatches:" (hybrid, cut, poly4).
CMP_SUMMARY = grep -E 'checked=|OVERALL|^Checked:|^Mismatches:|^PASS|^FAIL' || true

compare_cores_verify:
	@echo "=================== exp2 baseline ==================="
	@$(MAKE) --no-print-directory rtl_verify_pipelined     | $(CMP_SUMMARY)
	@echo "=================== exp2 optimised =================="
	@$(MAKE) --no-print-directory rtl_opt_verify_pipelined | $(CMP_SUMMARY)
	@echo "=================== exp2 width-optim ================"
	@$(MAKE) --no-print-directory rtl_optim_verify_pipelined | $(CMP_SUMMARY)
	@echo "=================== expe full LUT ==================="
	@$(MAKE) --no-print-directory rtl_expe_lut_verify_pipelined    | $(CMP_SUMMARY)
	@echo "=================== expe hybrid ====================="
	@$(MAKE) --no-print-directory rtl_expe_hybrid_verify_pipelined | $(CMP_SUMMARY)
	@echo "=================== expe cut ladder ================="
	@$(MAKE) --no-print-directory rtl_expe_cut_verify_pipelined    | $(CMP_SUMMARY)
	@echo "=================== expe poly4 ======================"
	@$(MAKE) --no-print-directory rtl_expe_poly4_verify_pipelined  | $(CMP_SUMMARY)
	@echo "=================== expe poly4 DSP =================="
	@$(MAKE) --no-print-directory rtl_expe_poly4_dsp_verify_pipelined | $(CMP_SUMMARY)

# Weryfikacja + dokladnosc ULP + synteza.
compare_all: compare_cores_verify ulp_compare compare_cores

# =========================================================================
# Rownowaznosc wszystkich implementacji (drop-in replacement)
#
# Wszystkie rdzenie sterowane jednym strumieniem wejsciowym, porownywane
# cykl po cyklu: dane, tvalid, tready oraz zmierzona latencja.
# NaN NIE sa pomijane - musza byc bit w bit identyczne.
# =========================================================================
EQUIV_TOP    = tests/bf16_expe_equiv_top.sv
EQUIV_TB     = tests/rtl_expe_equivalence_test.cpp
EQUIV_SRC    = $(RTL_DIR)/bf16_exp2_pkg.sv \
               $(RTL_DIR)/bf16_pipe_pad.sv \
               $(RTL_DIR)/bf16_decompose.sv \
               $(RTL_DIR)/bf16_recompose.sv \
               $(RTL_DIR)/bf16_early_out.sv \
               $(RTL_DIR)/bf16_log2e_mult.sv \
               $(RTL_DIR)/bf16_unified_shift.sv \
               $(RTL_DIR)/bf16_coeff_rom.sv \
               $(RTL_DIR)/bf16_linear_approx.sv \
               $(RTL_DIR)/bf16_normalize.sv \
               $(RTL_DIR)/bf16_round.sv \
               $(RTL_DIR)/bf16_exp2.sv \
               $(RTL_DIR)/bf16_expe_lut_rom.sv \
               $(RTL_DIR)/bf16_expe_lut.sv \
               $(RTL_DIR)/bf16_expe_sparse_decode.sv \
               $(RTL_DIR)/bf16_expe_hybrid_rom.sv \
               $(RTL_DIR)/bf16_expe_hybrid.sv \
               $(RTL_DIR)/bf16_expe_cut_rom.sv \
               $(RTL_DIR)/bf16_expe_cut.sv \
               $(RTL_DIR)/bf16_expe_poly4_rom.sv \
               $(RTL_DIR)/bf16_expe_poly4.sv \
               $(EQUIV_TOP)

.PHONY: rtl_equivalence_test
rtl_equivalence_test: gen_expe_lut_rom gen_expe_hybrid_tables gen_expe_cut_tables gen_expe_poly4_tables
	@echo "Building cross-implementation equivalence test ..."
	$(VERILATOR) -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
	    --top-module bf16_expe_equiv_top --cc -Mdir obj_dir_equiv \
	    $(VER_INC) $(EQUIV_SRC) --exe ../$(EQUIV_TB) $(VER_CFLAGS)
	make -j -C obj_dir_equiv -f Vbf16_expe_equiv_top.mk Vbf16_expe_equiv_top
	./obj_dir_equiv/Vbf16_expe_equiv_top

# =========================================================================
# RTL Verification -- full LUT exp(x) model (bf16_expe_lut)
# =========================================================================
LUT_TB       = tests/rtl_expe_lut_compliance_test.cpp
LUT_VER_FLAGS = -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
                --top-module bf16_expe_lut --cc
LUT_RTL_SRC  = $(RTL_DIR)/bf16_exp2_pkg.sv \
               $(RTL_DIR)/bf16_pipe_pad.sv \
               $(RTL_DIR)/bf16_decompose.sv \
               $(RTL_DIR)/bf16_early_out.sv \
               $(RTL_DIR)/bf16_expe_lut_rom.sv \
               $(RTL_DIR)/bf16_expe_lut.sv

.PHONY: gen_expe_lut_rom rtl_expe_lut_verify rtl_expe_lut_verify_pipelined

gen_expe_lut_rom: | $(BUILD_DIR)
	@echo "Generating bf16_expe_lut_rom.sv from LUT table ..."
	$(CXX) $(CXXFLAGS) -Isrc/approximations tests/gen_bf16_expe_lut_rom.cpp \
	    -o $(BUILD_DIR)/gen_expe_lut_rom
	./$(BUILD_DIR)/gen_expe_lut_rom

rtl_expe_lut_verify:
	@echo "Building combinational LUT RTL (REGISTER_STAGES=0) ..."
	$(VERILATOR) $(LUT_VER_FLAGS) -Mdir obj_dir_lut $(VER_INC) $(LUT_RTL_SRC) \
	    --exe $(LUT_TB) $(VER_CFLAGS)
	make -j -C obj_dir_lut -f Vbf16_expe_lut.mk Vbf16_expe_lut
	./obj_dir_lut/Vbf16_expe_lut

rtl_expe_lut_verify_pipelined:
	@echo "Building pipelined LUT RTL (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(LUT_VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_lut_pl $(VER_INC) $(LUT_RTL_SRC) \
	    --exe $(LUT_TB) $(VER_CFLAGS)
	make -j -C obj_dir_lut_pl -f Vbf16_expe_lut.mk Vbf16_expe_lut
	./obj_dir_lut_pl/Vbf16_expe_lut --latency 16

# =========================================================================
# Hybrid compressed-table exp(x): sparse thresholds + dense ROM
# =========================================================================
HYBRID_TB = tests/rtl_expe_hybrid_compliance_test.cpp
HYBRID_VER_FLAGS = -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
                   --top-module bf16_expe_hybrid --cc
HYBRID_RTL_SRC = $(RTL_DIR)/bf16_exp2_pkg.sv \
               $(RTL_DIR)/bf16_pipe_pad.sv \
                 $(RTL_DIR)/bf16_decompose.sv \
                 $(RTL_DIR)/bf16_early_out.sv \
                 $(RTL_DIR)/bf16_expe_sparse_decode.sv \
                 $(RTL_DIR)/bf16_expe_hybrid_rom.sv \
                 $(RTL_DIR)/bf16_expe_hybrid.sv

.PHONY: gen_expe_hybrid_tables expe_hybrid_test \
        rtl_expe_hybrid_verify rtl_expe_hybrid_verify_pipelined

gen_expe_hybrid_tables: | $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/gen_bf16_expe_hybrid_tables.cpp \
	    -o $(BUILD_DIR)/gen_bf16_expe_hybrid_tables
	./$(BUILD_DIR)/gen_bf16_expe_hybrid_tables

expe_hybrid_test:
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/bf16_expe_hybrid_test.cpp \
	    -o $(BUILD_DIR)/bf16_expe_hybrid_test
	./$(BUILD_DIR)/bf16_expe_hybrid_test

rtl_expe_hybrid_verify:
	@echo "Building combinational hybrid RTL (REGISTER_STAGES=0) ..."
	$(VERILATOR) $(HYBRID_VER_FLAGS) -Mdir obj_dir_hybrid $(VER_INC) $(HYBRID_RTL_SRC) \
	    --exe $(HYBRID_TB) $(VER_CFLAGS)
	make -j -C obj_dir_hybrid -f Vbf16_expe_hybrid.mk Vbf16_expe_hybrid
	./obj_dir_hybrid/Vbf16_expe_hybrid

rtl_expe_hybrid_verify_pipelined:
	@echo "Building pipelined hybrid RTL (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(HYBRID_VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_hybrid_pl \
	    $(VER_INC) $(HYBRID_RTL_SRC) --exe $(HYBRID_TB) $(VER_CFLAGS)
	make -j -C obj_dir_hybrid_pl -f Vbf16_expe_hybrid.mk Vbf16_expe_hybrid
	./obj_dir_hybrid_pl/Vbf16_expe_hybrid --latency 16

# =========================================================================
# Cut-point ladder exp(x): shared 2^-f ladder + candidate ROM
# =========================================================================
.PHONY: gen_expe_cut_tables expe_cut_test gen_expe_cut_approx

gen_expe_cut_tables: | $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 tests/gen_bf16_expe_cut_tables.cpp \
	    -o $(BUILD_DIR)/gen_bf16_expe_cut_tables
	./$(BUILD_DIR)/gen_bf16_expe_cut_tables

expe_cut_test: gen_expe_cut_tables
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 tests/bf16_expe_cut_test.cpp \
	    -o $(BUILD_DIR)/bf16_expe_cut_test
	./$(BUILD_DIR)/bf16_expe_cut_test

# Emit the golden-reference output of the cut-ladder and hybrid models,
# so ulp_error_analysis can score and cross-check them.
gen_expe_cut_approx: gen_expe_cut_tables
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 tests/gen_bf16_expe_cut_approx.cpp \
	    -o $(BUILD_DIR)/gen_bf16_expe_cut_approx
	./$(BUILD_DIR)/gen_bf16_expe_cut_approx

# Full comparison: regenerate every model's output, then score all of them.
.PHONY: ulp_compare
ulp_compare: gen_expe_lut_approx gen_expe_cut_approx gen_expe_poly4_approx \
             $(TARGET_ULP_ANALYSIS)
	./$(TARGET_ULP_ANALYSIS)

# -------------------------------------------------------------------------
# RTL verification of the cut-point ladder
# -------------------------------------------------------------------------
CUT_TB = tests/rtl_expe_cut_compliance_test.cpp
CUT_VER_FLAGS = -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
                --top-module bf16_expe_cut --cc
CUT_RTL_SRC = $(RTL_DIR)/bf16_exp2_pkg.sv \
               $(RTL_DIR)/bf16_pipe_pad.sv \
              $(RTL_DIR)/bf16_decompose.sv \
              $(RTL_DIR)/bf16_early_out.sv \
              $(RTL_DIR)/bf16_expe_cut_rom.sv \
              $(RTL_DIR)/bf16_expe_cut.sv

.PHONY: rtl_expe_cut_verify rtl_expe_cut_verify_pipelined rtl_expe_cut_all

rtl_expe_cut_verify: gen_expe_cut_tables
	@echo "Building combinational cut-ladder RTL (REGISTER_STAGES=0) ..."
	$(VERILATOR) $(CUT_VER_FLAGS) -Mdir obj_dir_cut $(VER_INC) $(CUT_RTL_SRC) \
	    --exe $(CUT_TB) $(VER_CFLAGS)
	make -j -C obj_dir_cut -f Vbf16_expe_cut.mk Vbf16_expe_cut
	./obj_dir_cut/Vbf16_expe_cut

rtl_expe_cut_verify_pipelined: gen_expe_cut_tables
	@echo "Building pipelined cut-ladder RTL (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(CUT_VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_cut_pl \
	    $(VER_INC) $(CUT_RTL_SRC) --exe $(CUT_TB) $(VER_CFLAGS)
	make -j -C obj_dir_cut_pl -f Vbf16_expe_cut.mk Vbf16_expe_cut
	./obj_dir_cut_pl/Vbf16_expe_cut --latency 16

rtl_expe_cut_all: rtl_expe_cut_verify rtl_expe_cut_verify_pipelined rtl_expe_cut_axi_test

.PHONY: rtl_expe_cut_axi_test

rtl_expe_cut_axi_test: gen_expe_cut_tables
	@echo "Building cut-ladder AXI-Stream protocol test (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(CUT_VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_cut_axi \
	    $(VER_INC) $(CUT_RTL_SRC) --exe tests/rtl_expe_cut_axi_test.cpp $(VER_CFLAGS)
	make -j -C obj_dir_cut_axi -f Vbf16_expe_cut.mk Vbf16_expe_cut
	./obj_dir_cut_axi/Vbf16_expe_cut

# =========================================================================
# Degree-4 minimax/Horner exp(x): DSP-heavy, storage-light counterpart of
# the cut-point ladder.  157 table bits, 4 DSP slices.
# =========================================================================
.PHONY: gen_expe_poly4_tables expe_poly4_test gen_expe_poly4_approx

gen_expe_poly4_tables: | $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 tests/gen_bf16_expe_poly4_tables.cpp \
	    -o $(BUILD_DIR)/gen_bf16_expe_poly4_tables
	./$(BUILD_DIR)/gen_bf16_expe_poly4_tables

expe_poly4_test: gen_expe_poly4_tables gen_expe_cut_tables
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 tests/bf16_expe_poly4_test.cpp \
	    -o $(BUILD_DIR)/bf16_expe_poly4_test
	./$(BUILD_DIR)/bf16_expe_poly4_test

# Golden-reference output of the degree-4 model, for ulp_error_analysis.
gen_expe_poly4_approx: gen_expe_poly4_tables
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O2 tests/gen_bf16_expe_poly4_approx.cpp \
	    -o $(BUILD_DIR)/gen_bf16_expe_poly4_approx
	./$(BUILD_DIR)/gen_bf16_expe_poly4_approx

# -------------------------------------------------------------------------
# RTL verification of the degree-4 Horner variant
# -------------------------------------------------------------------------
POLY4_TB = tests/rtl_expe_poly4_compliance_test.cpp
POLY4_VER_FLAGS = -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
                  --top-module bf16_expe_poly4 --cc
POLY4_RTL_SRC = $(RTL_DIR)/bf16_exp2_pkg.sv \
               $(RTL_DIR)/bf16_pipe_pad.sv \
                $(RTL_DIR)/bf16_decompose.sv \
                $(RTL_DIR)/bf16_early_out.sv \
                $(RTL_DIR)/bf16_expe_poly4_rom.sv \
                $(RTL_DIR)/bf16_expe_poly4.sv

.PHONY: rtl_expe_poly4_verify rtl_expe_poly4_verify_pipelined \
        rtl_expe_poly4_axi_test rtl_expe_poly4_all

rtl_expe_poly4_verify: gen_expe_poly4_tables
	@echo "Building combinational degree-4 RTL (REGISTER_STAGES=0) ..."
	$(VERILATOR) $(POLY4_VER_FLAGS) -Mdir obj_dir_poly4 $(VER_INC) \
	    $(POLY4_RTL_SRC) --exe $(POLY4_TB) $(VER_CFLAGS)
	make -j -C obj_dir_poly4 -f Vbf16_expe_poly4.mk Vbf16_expe_poly4
	./obj_dir_poly4/Vbf16_expe_poly4

rtl_expe_poly4_verify_pipelined: gen_expe_poly4_tables
	@echo "Building pipelined degree-4 RTL (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(POLY4_VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_poly4_pl \
	    $(VER_INC) $(POLY4_RTL_SRC) --exe $(POLY4_TB) $(VER_CFLAGS)
	make -j -C obj_dir_poly4_pl -f Vbf16_expe_poly4.mk Vbf16_expe_poly4
	./obj_dir_poly4_pl/Vbf16_expe_poly4 --latency 16

rtl_expe_poly4_axi_test: gen_expe_poly4_tables
	@echo "Building degree-4 AXI-Stream protocol test (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(POLY4_VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_poly4_axi \
	    $(VER_INC) $(POLY4_RTL_SRC) --exe tests/rtl_expe_poly4_axi_test.cpp \
	    $(VER_CFLAGS)
	make -j -C obj_dir_poly4_axi -f Vbf16_expe_poly4.mk Vbf16_expe_poly4
	./obj_dir_poly4_axi/Vbf16_expe_poly4

# -------------------------------------------------------------------------
# DSP front-end variant: the barrel shifter and the CSD constant multiply are
# replaced by three more DSP slices (7 DSP total).  Bit-identical output, one
# extra pipeline stage.
# -------------------------------------------------------------------------
.PHONY: rtl_expe_poly4_dsp_verify rtl_expe_poly4_dsp_verify_pipelined \
        rtl_expe_poly4_dsp_axi_test rtl_expe_poly4_dsp_all

rtl_expe_poly4_dsp_verify: gen_expe_poly4_tables
	@echo "Building combinational DSP front-end (DSP_FRONTEND=1) ..."
	$(VERILATOR) $(POLY4_VER_FLAGS) -GDSP_FRONTEND=1 -Mdir obj_dir_poly4_dsp \
	    $(VER_INC) $(POLY4_RTL_SRC) --exe $(POLY4_TB) $(VER_CFLAGS)
	make -j -C obj_dir_poly4_dsp -f Vbf16_expe_poly4.mk Vbf16_expe_poly4
	./obj_dir_poly4_dsp/Vbf16_expe_poly4

rtl_expe_poly4_dsp_verify_pipelined: gen_expe_poly4_tables
	@echo "Building pipelined DSP front-end (DSP_FRONTEND=1, latency 8) ..."
	$(VERILATOR) $(POLY4_VER_FLAGS) -GREGISTER_STAGES=1 -GDSP_FRONTEND=1 \
	    -Mdir obj_dir_poly4_dsp_pl $(VER_INC) $(POLY4_RTL_SRC) \
	    --exe $(POLY4_TB) $(VER_CFLAGS)
	make -j -C obj_dir_poly4_dsp_pl -f Vbf16_expe_poly4.mk Vbf16_expe_poly4
	./obj_dir_poly4_dsp_pl/Vbf16_expe_poly4 --latency 16

rtl_expe_poly4_dsp_axi_test: gen_expe_poly4_tables
	@echo "Building DSP front-end AXI-Stream protocol test ..."
	$(VERILATOR) $(POLY4_VER_FLAGS) -GREGISTER_STAGES=1 -GDSP_FRONTEND=1 \
	    -Mdir obj_dir_poly4_dsp_axi $(VER_INC) $(POLY4_RTL_SRC) \
	    --exe tests/rtl_expe_poly4_axi_test.cpp $(VER_CFLAGS)
	make -j -C obj_dir_poly4_dsp_axi -f Vbf16_expe_poly4.mk Vbf16_expe_poly4
	./obj_dir_poly4_dsp_axi/Vbf16_expe_poly4

rtl_expe_poly4_dsp_all: rtl_expe_poly4_dsp_verify \
                        rtl_expe_poly4_dsp_verify_pipelined \
                        rtl_expe_poly4_dsp_axi_test

# -------------------------------------------------------------------------
# Vivado synthesis.  VIVADO can be overridden if it is not on PATH.
# -------------------------------------------------------------------------
VIVADO ?= $(HOME)/AMD/2025.2/Vivado/bin/vivado

.PHONY: synth_expe_poly4_dsp sweep_expe_poly4

synth_expe_poly4_dsp: gen_expe_poly4_tables
	$(VIVADO) -mode batch -nojournal -nolog \
	    -source scripts/synth_expe_poly4_dsp.tcl

# Synthesizes all four front-end / reset combinations and prints the
# comparison table that backs the resource numbers in the docs.
sweep_expe_poly4: gen_expe_poly4_tables
	$(VIVADO) -mode batch -nojournal -nolog \
	    -source scripts/sweep_poly4_configs.tcl

rtl_expe_poly4_all: rtl_expe_poly4_verify rtl_expe_poly4_verify_pipelined \
                    rtl_expe_poly4_axi_test rtl_expe_poly4_dsp_all


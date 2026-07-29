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
TEST_SRC_GEN_EXPE_LUT = $(TEST_DIR)/gen_bf16_expe_lut.cpp
TEST_SRC_EXPE_LUT_TEST = $(TEST_DIR)/bf16_expe_lut_test.cpp
TEST_SRC_GEN_EXPE_LUT_APPROX = $(TEST_DIR)/gen_bf16_expe_lut_approx.cpp

# Default rule: build all
.PHONY: all run run_exhaustive gen_approx ulp_analysis run_linear_approx gen_packed rne_sweep clean

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

# Generate the full expe LUT header, then build & run the ULP verification test.
gen_expe_lut: $(TARGET_GEN_EXPE_LUT)
	./$(TARGET_GEN_EXPE_LUT)

expe_lut_test: gen_expe_lut $(TARGET_EXPE_LUT_TEST)
	./$(TARGET_EXPE_LUT_TEST)

run_linear_approx: $(TARGET_LINEAR_APPROX)
	./$(TARGET_LINEAR_APPROX)

clean:
	rm -rf $(BUILD_DIR)

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
	./obj_dir_pl/Vbf16_exp2 --latency 12

rtl_axi_test:
	@echo "Building AXI-Stream protocol test (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_axi $(VER_INC) $(RTL_SRC) \
	    --exe tests/rtl_axi_stream_test.cpp $(VER_CFLAGS)
	make -j -C obj_dir_axi -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_axi/Vbf16_exp2

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
	./obj_dir_dsps_pl/Vbf16_exp2 --latency 12

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
	./obj_dir_opt_pl/Vbf16_exp2 --latency 12

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

# =========================================================================
# Porownanie wszystkich implementacji obok siebie
# =========================================================================
.PHONY: compare_cores compare_cores_verify compare_all

# Synteza wszystkich rdzeni w jednym przebiegu -> jedna tabela zasobow.
compare_cores: gen_expe_lut_rom gen_expe_hybrid_tables gen_expe_cut_tables gen_expe_poly4_tables
	$(VIVADO) -mode batch -nojournal -nolog -source scripts/compare_all_cores.tcl

# Weryfikacja funkcjonalna wszystkich rdzeni (Verilator, wyczerpujaca).
# Rdzenie maja rozne testbenche, wiec podsumowanie filtrujemy po obu formatach:
# "checked=/OVERALL" (exp2, lut) oraz "Checked:/Mismatches:" (hybrid, cut, poly4).
CMP_SUMMARY = grep -E 'checked=|OVERALL|^Checked:|^Mismatches:|^PASS|^FAIL' || true

compare_cores_verify:
	@echo "=================== exp2 baseline ==================="
	@$(MAKE) --no-print-directory rtl_verify_pipelined     | $(CMP_SUMMARY)
	@echo "=================== exp2 optimised =================="
	@$(MAKE) --no-print-directory rtl_opt_verify_pipelined | $(CMP_SUMMARY)
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
	./obj_dir_lut_pl/Vbf16_expe_lut --latency 12

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
	./obj_dir_hybrid_pl/Vbf16_expe_hybrid --latency 12

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
	./obj_dir_cut_pl/Vbf16_expe_cut --latency 12

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
	./obj_dir_poly4_pl/Vbf16_expe_poly4 --latency 12

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
	./obj_dir_poly4_dsp_pl/Vbf16_expe_poly4 --latency 12

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


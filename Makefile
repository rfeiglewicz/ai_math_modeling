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
	./obj_dir_pl/Vbf16_exp2 --latency 6

rtl_axi_test:
	@echo "Building AXI-Stream protocol test (REGISTER_STAGES=1) ..."
	$(VERILATOR) $(VER_FLAGS) -GREGISTER_STAGES=1 -Mdir obj_dir_axi $(VER_INC) $(RTL_SRC) \
	    --exe tests/rtl_axi_stream_test.cpp $(VER_CFLAGS)
	make -j -C obj_dir_axi -f Vbf16_exp2.mk Vbf16_exp2
	./obj_dir_axi/Vbf16_exp2

# =========================================================================
# RTL Verification -- full LUT exp(x) model (bf16_expe_lut)
# =========================================================================
LUT_TB       = tests/rtl_expe_lut_compliance_test.cpp
LUT_VER_FLAGS = -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-LATCH \
                --top-module bf16_expe_lut --cc
LUT_RTL_SRC  = $(RTL_DIR)/bf16_exp2_pkg.sv \
               $(RTL_DIR)/bf16_decompose.sv \
               $(RTL_DIR)/bf16_early_out.sv \
               $(RTL_DIR)/bf16_expe_lut_rom.sv \
               $(RTL_DIR)/bf16_expe_lut.sv

.PHONY: gen_expe_lut_rom rtl_expe_lut_verify rtl_expe_lut_verify_pipelined

gen_expe_lut_rom:
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
	./obj_dir_lut_pl/Vbf16_expe_lut --latency 4
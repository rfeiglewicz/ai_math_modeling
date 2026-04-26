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

# Source files
TEST_SRC_MAIN = $(TEST_DIR)/fp_utils_test.cpp
TEST_SRC_EXHAUSTIVE = $(TEST_DIR)/fp_utils_exhaustive_test.cpp
TEST_SRC_GEN_APPROX = $(TEST_DIR)/gen_bf16_exp2_approx.cpp
TEST_SRC_ULP_ANALYSIS = $(TEST_DIR)/ulp_error_analysis.cpp
TEST_SRC_LINEAR_APPROX = $(TEST_DIR)/test_bf16_linear_approx.cpp
SRC_GEN_PACKED = modeling/coeff_gen/gen_packed_coeffs.cpp

# Default rule: build all
.PHONY: all run run_exhaustive gen_approx ulp_analysis run_linear_approx gen_packed clean

all: $(TARGET_MAIN) $(TARGET_EXHAUSTIVE) $(TARGET_GEN_APPROX) $(TARGET_ULP_ANALYSIS) $(TARGET_LINEAR_APPROX) $(TARGET_GEN_PACKED)

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

# Run rules
run: $(TARGET_MAIN)
	./$(TARGET_MAIN)

run_exhaustive: $(TARGET_EXHAUSTIVE)
	./$(TARGET_EXHAUSTIVE)

gen_approx: $(TARGET_GEN_APPROX)
	./$(TARGET_GEN_APPROX)

ulp_analysis: $(TARGET_ULP_ANALYSIS)
	./$(TARGET_ULP_ANALYSIS)

gen_packed: $(TARGET_GEN_PACKED)
	./$(TARGET_GEN_PACKED)

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

.PHONY: rtl_verify rtl_verify_pipelined verilate

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
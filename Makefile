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
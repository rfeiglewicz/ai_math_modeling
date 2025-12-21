# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./src/utils

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

# Source files
TEST_SRC_MAIN = $(TEST_DIR)/fp_utils_test.cpp
TEST_SRC_EXHAUSTIVE = $(TEST_DIR)/fp_utils_exhaustive_test.cpp
TEST_SRC_GEN_APPROX = $(TEST_DIR)/gen_bf16_exp2_approx.cpp
TEST_SRC_ULP_ANALYSIS = $(TEST_DIR)/ulp_error_analysis.cpp
TEST_SRC_LINEAR_APPROX = $(TEST_DIR)/test_bf16_linear_approx.cpp

# Default rule: build all
all: $(TARGET_MAIN) $(TARGET_EXHAUSTIVE) $(TARGET_GEN_APPROX) $(TARGET_ULP_ANALYSIS) $(TARGET_LINEAR_APPROX)

# Rule to link the main test executable
$(TARGET_MAIN): $(TEST_SRC_MAIN)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to link the exhaustive test executable
$(TARGET_EXHAUSTIVE): $(TEST_SRC_EXHAUSTIVE)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to link the approximation generator
$(TARGET_GEN_APPROX): $(TEST_SRC_GEN_APPROX)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to link the ULP analysis test
$(TARGET_ULP_ANALYSIS): $(TEST_SRC_ULP_ANALYSIS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to link the linear approximation test
$(TARGET_LINEAR_APPROX): $(TEST_SRC_LINEAR_APPROX)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Run standard tests
run: $(TARGET_MAIN)
	./$(TARGET_MAIN)

# Run exhaustive tests
run_exhaustive: $(TARGET_EXHAUSTIVE)
	./$(TARGET_EXHAUSTIVE)

# Run approximation generator
gen_approx: $(TARGET_GEN_APPROX)
	./$(TARGET_GEN_APPROX)

# Run ULP error analysis
ulp_analysis: $(TARGET_ULP_ANALYSIS)
	./$(TARGET_ULP_ANALYSIS)

# Run linear approximation test
run_linear_approx: $(TARGET_LINEAR_APPROX)
	./$(TARGET_LINEAR_APPROX)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run run_exhaustive gen_approx ulp_analysis run_linear_approx clean
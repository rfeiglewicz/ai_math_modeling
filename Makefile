# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./src/utils

# Directories
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

# Target executable name
TARGET = $(BUILD_DIR)/fp_utils_test

# Source files
TEST_SRC = $(TEST_DIR)/fp_utils_test.cpp

# Default rule
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to run the test
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run clean
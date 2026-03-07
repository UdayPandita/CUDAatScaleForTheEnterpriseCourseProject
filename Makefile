# =============================================================================
# Makefile for CUDA Aerial Image Processing Pipeline
# =============================================================================
# 
# This Makefile compiles the CUDA image processing project.
# Requires: NVIDIA CUDA Toolkit (nvcc compiler)
#
# Usage:
#   make          - Build the project (release mode)
#   make debug    - Build with debug symbols
#   make clean    - Remove all build artifacts
#   make run      - Build and run with default settings
#   make help     - Show this help message
#
# =============================================================================

# Compiler settings
NVCC        := nvcc
CXX         := g++

# Project directories
SRC_DIR     := src
BUILD_DIR   := build
BIN_DIR     := bin

# Target executable
TARGET      := $(BIN_DIR)/cuda_image_processor

# Source files
CU_SOURCES  := $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# Object files
CU_OBJECTS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
OBJECTS     := $(CU_OBJECTS) $(CPP_OBJECTS)

# CUDA architecture flags
# Supports common GPU architectures (modify based on your GPU)
# sm_50: Maxwell (GTX 900 series)
# sm_60: Pascal (GTX 1000 series)
# sm_70: Volta (V100)
# sm_75: Turing (RTX 2000 series)
# sm_80: Ampere (RTX 3000 series)
# sm_86: Ampere (RTX 3000 mobile)
# sm_89: Ada Lovelace (RTX 4000 series)
CUDA_ARCH   := -gencode arch=compute_50,code=sm_50 \
               -gencode arch=compute_60,code=sm_60 \
               -gencode arch=compute_70,code=sm_70 \
               -gencode arch=compute_75,code=sm_75 \
               -gencode arch=compute_80,code=sm_80 \
               -gencode arch=compute_86,code=sm_86

# Compiler flags
NVCC_FLAGS  := -std=c++17 $(CUDA_ARCH) -Xcompiler -Wall
CXX_FLAGS   := -std=c++17 -Wall -Wextra

# Include paths
INCLUDES    := -I$(SRC_DIR)

# Libraries
LIBS        := -lcudart

# Release optimization flags
RELEASE_FLAGS := -O3 -DNDEBUG

# Debug flags
DEBUG_FLAGS := -g -G -O0 -DDEBUG

# Default build mode
BUILD_MODE  ?= release

ifeq ($(BUILD_MODE),debug)
    NVCC_FLAGS += $(DEBUG_FLAGS)
    CXX_FLAGS  += -g -O0 -DDEBUG
else
    NVCC_FLAGS += $(RELEASE_FLAGS)
    CXX_FLAGS  += -O3 -DNDEBUG
endif

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean debug release run help directories

# Default target
all: directories $(TARGET)
	@echo "Build complete: $(TARGET)"

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Link all objects into final executable
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling C++: $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# Debug build
debug:
	$(MAKE) BUILD_MODE=debug all

# Release build
release:
	$(MAKE) BUILD_MODE=release all

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete."

# Run with default settings
run: all
	@echo "Running CUDA Image Processor..."
	./$(TARGET) -i ./aerials -o ./output -m all -v

# Run with 10 images only
run-10: all
	@echo "Running CUDA Image Processor (10 images)..."
	./$(TARGET) -i ./aerials -o ./output -m all -n 10 -v

# Run edge detection only
run-edge: all
	@echo "Running edge detection..."
	./$(TARGET) -i ./aerials -o ./output -m edge -v

# Show help
help:
	@echo "============================================================"
	@echo "CUDA Aerial Image Processing Pipeline - Build System"
	@echo "============================================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build the project (release mode)"
	@echo "  make debug        - Build with debug symbols"
	@echo "  make release      - Build optimized release version"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make run          - Build and run with all images"
	@echo "  make run-10       - Build and run with 10 images"
	@echo "  make run-edge     - Build and run edge detection only"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Build output:"
	@echo "  Executable: $(TARGET)"
	@echo "  Objects:    $(BUILD_DIR)/"
	@echo ""
	@echo "Requirements:"
	@echo "  - NVIDIA CUDA Toolkit (nvcc)"
	@echo "  - C++17 compatible compiler"
	@echo "  - CUDA-capable GPU"
	@echo ""

# Dependencies
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu $(SRC_DIR)/image_kernels.cuh $(SRC_DIR)/image_utils.h
$(BUILD_DIR)/image_utils.o: $(SRC_DIR)/image_utils.cpp $(SRC_DIR)/image_utils.h $(SRC_DIR)/stb_image.h $(SRC_DIR)/stb_image_write.h

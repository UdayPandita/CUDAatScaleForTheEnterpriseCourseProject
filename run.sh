#!/bin/bash
# =============================================================================
# run.sh - CUDA Aerial Image Processing Pipeline Runner
# =============================================================================
#
# This script builds and runs the CUDA image processing pipeline.
# It provides a convenient interface for common operations.
#
# Usage: ./run.sh [OPTIONS]
#
# Options:
#   --build         Build the project only
#   --clean         Clean build artifacts
#   --all           Process all images with all filters (default)
#   --edge          Process with edge detection only
#   --blur          Process with blur only
#   --grayscale     Process with grayscale conversion only
#   --enhance       Process with image enhancement only
#   --num N         Process only N images
#   --verbose       Enable verbose output
#   --help          Show this help message
#
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="$SCRIPT_DIR/bin/cuda_image_processor"
INPUT_DIR="$SCRIPT_DIR/aerials"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Default options
MODE="all"
NUM_IMAGES=""
VERBOSE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
print_banner() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║     CUDA Aerial Image Processing Pipeline                     ║"
    echo "║     GPU-Accelerated Image Processing for Large Datasets       ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
}

# Show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build         Build the project only"
    echo "  --clean         Clean build artifacts"
    echo "  --all           Process all images with all filters (default)"
    echo "  --edge          Process with edge detection only"
    echo "  --blur          Process with blur only"
    echo "  --grayscale     Process with grayscale conversion only"
    echo "  --enhance       Process with image enhancement only"
    echo "  --num N         Process only N images"
    echo "  --verbose       Enable verbose output"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Build and run with all processing"
    echo "  $0 --edge               # Run edge detection only"
    echo "  $0 --num 10 --verbose   # Process 10 images with verbose output"
    echo "  $0 --build              # Build only, don't run"
    echo "  $0 --clean              # Clean build artifacts"
    echo ""
}

# Check for CUDA
check_cuda() {
    if ! command -v nvcc &> /dev/null; then
        print_error "NVIDIA CUDA Toolkit not found!"
        print_error "Please install CUDA Toolkit and ensure 'nvcc' is in your PATH"
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)
    print_info "CUDA Version: $CUDA_VERSION"
}

# Check for GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        if [ -n "$GPU_NAME" ]; then
            print_info "GPU Detected: $GPU_NAME"
        else
            print_warning "No GPU detected (nvidia-smi found but no GPU reported)"
        fi
    else
        print_warning "nvidia-smi not found - cannot verify GPU"
    fi
}

# Build project
build_project() {
    print_info "Building project..."
    cd "$SCRIPT_DIR"
    
    if make; then
        print_success "Build completed successfully!"
    else
        print_error "Build failed!"
        exit 1
    fi
}

# Clean project
clean_project() {
    print_info "Cleaning build artifacts..."
    cd "$SCRIPT_DIR"
    make clean
    print_success "Clean completed!"
}

# Run the processor
run_processor() {
    if [ ! -f "$EXECUTABLE" ]; then
        print_error "Executable not found! Building first..."
        build_project
    fi
    
    # Check input directory
    if [ ! -d "$INPUT_DIR" ]; then
        print_error "Input directory not found: $INPUT_DIR"
        exit 1
    fi
    
    # Count input images
    IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -name "*.tiff" -o -name "*.tif" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" \) | wc -l)
    print_info "Found $IMAGE_COUNT images in $INPUT_DIR"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Build command
    CMD="$EXECUTABLE -i $INPUT_DIR -o $OUTPUT_DIR -m $MODE"
    
    if [ -n "$NUM_IMAGES" ]; then
        CMD="$CMD -n $NUM_IMAGES"
    fi
    
    if [ -n "$VERBOSE" ]; then
        CMD="$CMD -v"
    fi
    
    print_info "Running: $CMD"
    echo ""
    
    # Run with timing
    START_TIME=$(date +%s.%N)
    
    if $CMD; then
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        
        echo ""
        print_success "Processing completed in ${ELAPSED}s"
        print_info "Output saved to: $OUTPUT_DIR"
        
        # Count output files
        OUTPUT_COUNT=$(find "$OUTPUT_DIR" -type f -name "*.png" | wc -l)
        print_info "Generated $OUTPUT_COUNT output images"
    else
        print_error "Processing failed!"
        exit 1
    fi
}

# Main
main() {
    print_banner
    
    # Parse arguments
    BUILD_ONLY=false
    CLEAN_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_ONLY=true
                shift
                ;;
            --clean)
                CLEAN_ONLY=true
                shift
                ;;
            --all)
                MODE="all"
                shift
                ;;
            --edge)
                MODE="edge"
                shift
                ;;
            --blur)
                MODE="blur"
                shift
                ;;
            --grayscale)
                MODE="grayscale"
                shift
                ;;
            --enhance)
                MODE="enhance"
                shift
                ;;
            --num)
                NUM_IMAGES="$2"
                shift 2
                ;;
            --verbose|-v)
                VERBOSE="-v"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Execute requested action
    if [ "$CLEAN_ONLY" = true ]; then
        clean_project
        exit 0
    fi
    
    # Check environment
    check_cuda
    check_gpu
    echo ""
    
    if [ "$BUILD_ONLY" = true ]; then
        build_project
        exit 0
    fi
    
    # Build and run
    build_project
    echo ""
    run_processor
}

# Run main
main "$@"

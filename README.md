# CUDA Aerial Image Processing

## Project Description

This project processes aerial images using GPU-accelerated CUDA kernels. It applies various image processing operations including grayscale conversion, Gaussian blur, Sobel edge detection, and image enhancement. The processing is implemented using custom CUDA kernels for parallel execution on NVIDIA GPUs.

## Quickstart

To run this, you must have access to an NVIDIA GPU and CUDA toolkit. Clone this repository and build:

```bash
git clone https://github.com/UdayPandita/CUDAatScaleForTheEnterpriseCourseProject.git
cd CUDAatScaleForTheEnterpriseCourseProject
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Run the Pipeline

```bash
./bin/cuda_image_processor -i ../aerials -o ../output -m all
```

On Windows:
```bash
.\bin\Release\cuda_image_processor.exe -i ..\aerials -o ..\output -m all
```

## Process Specific Mode

```bash
./bin/cuda_image_processor -i ../aerials -o ../output -m edge
./bin/cuda_image_processor -i ../aerials -o ../output -m blur
./bin/cuda_image_processor -i ../aerials -o ../output -m grayscale
```

## Code Organization

`build/bin/` This folder holds all binary/executable code that is built automatically.

`aerials/` This folder holds the USC-SIPI aerial image dataset (38 TIFF images).

`src/` The source code for this project.

`README.md` The description of the project.

`CMakeLists.txt` CMake build configuration for this project.

`run.sh` Convenience script for running the pipeline (Linux/macOS).

`docs/` Contains sample execution log for proof of execution.

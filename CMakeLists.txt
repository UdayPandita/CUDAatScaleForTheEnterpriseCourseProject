cmake_minimum_required(VERSION 3.18)
project(CUDAImageProcessing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(SOURCES
    src/main.cu
    src/image_utils.cpp
)

set(HEADERS
    src/image_kernels.cuh
    src/image_utils.h
    src/stb_image.h
    src/stb_image_write.h
)

add_executable(cuda_image_processor ${SOURCES} ${HEADERS})

target_include_directories(cuda_image_processor PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(WIN32)
    target_compile_definitions(cuda_image_processor PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(cuda_image_processor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_SEPARABLE_COMPILATION ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_image_processor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
else()
    target_compile_options(cuda_image_processor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

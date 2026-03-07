#ifndef IMAGE_KERNELS_CUH_
#define IMAGE_KERNELS_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void GrayscaleKernel(const unsigned char* input, 
                                 unsigned char* output,
                                 int width, 
                                 int height, 
                                 int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = y * width + x;
    int input_idx = pixel_idx * channels;
    
    float gray = 0.299f * input[input_idx] +
                 0.587f * input[input_idx + 1] +
                 0.114f * input[input_idx + 2];
    
    output[pixel_idx] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, gray)));
}

__global__ void GaussianBlurKernel(const unsigned char* input, 
                                    unsigned char* output,
                                    int width, 
                                    int height, 
                                    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sigma = radius / 2.0f;
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            
            float weight = expf(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
            
            sum += weight * input[ny * width + nx];
            weight_sum += weight;
        }
    }
    
    output[y * width + x] = static_cast<unsigned char>(sum / weight_sum);
}

__global__ void SobelEdgeKernel(const unsigned char* input, 
                                 unsigned char* output,
                                 int width, 
                                 int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }
    
    int gx = 0, gy = 0;
    
    gx += -1 * input[(y - 1) * width + (x - 1)];
    gx +=  0 * input[(y - 1) * width + x];
    gx +=  1 * input[(y - 1) * width + (x + 1)];
    
    gy += -1 * input[(y - 1) * width + (x - 1)];
    gy += -2 * input[(y - 1) * width + x];
    gy += -1 * input[(y - 1) * width + (x + 1)];
    
    gx += -2 * input[y * width + (x - 1)];
    gx +=  0 * input[y * width + x];
    gx +=  2 * input[y * width + (x + 1)];
    
    gy +=  0 * input[y * width + (x - 1)];
    gy +=  0 * input[y * width + x];
    gy +=  0 * input[y * width + (x + 1)];
    
    gx += -1 * input[(y + 1) * width + (x - 1)];
    gx +=  0 * input[(y + 1) * width + x];
    gx +=  1 * input[(y + 1) * width + (x + 1)];
    
    gy +=  1 * input[(y + 1) * width + (x - 1)];
    gy +=  2 * input[(y + 1) * width + x];
    gy +=  1 * input[(y + 1) * width + (x + 1)];
    
    float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
    
    output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, magnitude));
}

__global__ void EnhanceKernel(const unsigned char* input, 
                               unsigned char* output,
                               int width, 
                               int height, 
                               int channels,
                               float contrast, 
                               int brightness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; ++c) {
        float value = input[pixel_idx + c];
        value = (value - 128.0f) * contrast + 128.0f + brightness;
        value = fminf(255.0f, fmaxf(0.0f, value));
        output[pixel_idx + c] = static_cast<unsigned char>(value);
    }
}

__global__ void InvertKernel(const unsigned char* input, 
                              unsigned char* output,
                              int width, 
                              int height, 
                              int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; ++c) {
        output[pixel_idx + c] = 255 - input[pixel_idx + c];
    }
}

__global__ void ThresholdKernel(const unsigned char* input, 
                                 unsigned char* output,
                                 int width, 
                                 int height, 
                                 unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    output[idx] = (input[idx] >= threshold) ? 255 : 0;
}

__global__ void HistogramEqualizationKernel(const unsigned char* input, 
                                             unsigned char* output,
                                             const unsigned char* lut,
                                             int width, 
                                             int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    output[idx] = lut[input[idx]];
}

__global__ void HistogramKernel(const unsigned char* input, 
                                 unsigned int* histogram,
                                 int width, 
                                 int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    atomicAdd(&histogram[input[idx]], 1);
}

__global__ void BoxBlurKernel(const unsigned char* input, 
                               unsigned char* output,
                               int width, 
                               int height, 
                               int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int sum = 0;
    int count = 0;
    
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            
            sum += input[ny * width + nx];
            count++;
        }
    }
    
    output[y * width + x] = static_cast<unsigned char>(sum / count);
}

__global__ void SharpenKernel(const unsigned char* input, 
                               const unsigned char* blurred,
                               unsigned char* output,
                               int width, 
                               int height, 
                               float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float sharpened = input[idx] + amount * (input[idx] - blurred[idx]);
    
    output[idx] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sharpened)));
}

#endif

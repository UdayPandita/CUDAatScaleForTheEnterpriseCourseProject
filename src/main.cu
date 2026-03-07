#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <fstream>

#include "image_kernels.cuh"
#include "image_utils.h"

namespace fs = std::filesystem;

void PrintUsage(const char* program_name) {
    std::cout << "CUDA Aerial Image Processing Pipeline\n";
    std::cout << "======================================\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input <path>     Input directory containing images (required)\n";
    std::cout << "  -o, --output <path>    Output directory for processed images (default: ./output)\n";
    std::cout << "  -m, --mode <mode>      Processing mode (default: all)\n";
    std::cout << "                         Options: grayscale, blur, edge, enhance, all\n";
    std::cout << "  -n, --num-images <n>   Number of images to process (default: all)\n";
    std::cout << "  -b, --blur-radius <r>  Gaussian blur radius (default: 3)\n";
    std::cout << "  -v, --verbose          Enable verbose output\n";
    std::cout << "  -h, --help             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -i ./aerials -o ./output -m all\n";
    std::cout << "  " << program_name << " -i ./aerials -m edge -n 10\n";
    std::cout << "  " << program_name << " -i ./aerials --blur-radius 5 --verbose\n";
}

struct ProgramArgs {
    std::string input_dir = "";
    std::string output_dir = "./output";
    std::string mode = "all";
    int num_images = -1;
    int blur_radius = 3;
    bool verbose = false;
};

bool ParseArguments(int argc, char* argv[], ProgramArgs& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return false;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                args.input_dir = argv[++i];
            } else {
                std::cerr << "Error: --input requires a path argument\n";
                return false;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                args.output_dir = argv[++i];
            } else {
                std::cerr << "Error: --output requires a path argument\n";
                return false;
            }
        } else if (arg == "-m" || arg == "--mode") {
            if (i + 1 < argc) {
                args.mode = argv[++i];
                if (args.mode != "grayscale" && args.mode != "blur" && 
                    args.mode != "edge" && args.mode != "enhance" && args.mode != "all") {
                    std::cerr << "Error: Invalid mode '" << args.mode << "'\n";
                    return false;
                }
            } else {
                std::cerr << "Error: --mode requires a mode argument\n";
                return false;
            }
        } else if (arg == "-n" || arg == "--num-images") {
            if (i + 1 < argc) {
                args.num_images = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --num-images requires a number argument\n";
                return false;
            }
        } else if (arg == "-b" || arg == "--blur-radius") {
            if (i + 1 < argc) {
                args.blur_radius = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --blur-radius requires a number argument\n";
                return false;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            PrintUsage(argv[0]);
            return false;
        }
    }
    
    if (args.input_dir.empty()) {
        std::cerr << "Error: Input directory is required\n";
        PrintUsage(argv[0]);
        return false;
    }
    
    return true;
}

std::vector<std::string> GetImageFiles(const std::string& dir_path, int max_images = -1) {
    std::vector<std::string> image_files;
    std::vector<std::string> extensions = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"};
    
    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                for (const auto& valid_ext : extensions) {
                    if (ext == valid_ext) {
                        image_files.push_back(entry.path().string());
                        break;
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
        return {};
    }
    
    std::sort(image_files.begin(), image_files.end());
    
    if (max_images > 0 && static_cast<size_t>(max_images) < image_files.size()) {
        image_files.resize(max_images);
    }
    
    return image_files;
}

double ProcessImage(const std::string& input_path, 
                    const std::string& output_dir,
                    const std::string& mode,
                    int blur_radius,
                    bool verbose) {
    int width, height, channels;
    unsigned char* h_input = LoadImage(input_path.c_str(), &width, &height, &channels);
    
    if (h_input == nullptr) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return -1.0;
    }
    
    if (verbose) {
        std::cout << "  Loaded: " << width << "x" << height << " (" << channels << " channels)\n";
    }
    
    std::string filename = fs::path(input_path).stem().string();
    
    size_t image_size = width * height * channels * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);
    
    unsigned char *d_input, *d_gray, *d_output;
    float *d_temp;
    
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_gray, gray_size);
    cudaMalloc(&d_output, image_size);
    cudaMalloc(&d_temp, width * height * sizeof(float));
    
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    unsigned char* h_gray = new unsigned char[width * height];
    unsigned char* h_output = new unsigned char[image_size];
    
    if (mode == "grayscale" || mode == "all") {
        GrayscaleKernel<<<grid_size, block_size>>>(d_input, d_gray, width, height, channels);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);
        
        std::string out_path = output_dir + "/" + filename + "_grayscale.png";
        SaveImage(out_path.c_str(), h_gray, width, height, 1);
        
        if (verbose) {
            std::cout << "  Saved grayscale: " << out_path << "\n";
        }
    }
    
    if (mode == "blur" || mode == "all") {
        if (mode != "all") {
            GrayscaleKernel<<<grid_size, block_size>>>(d_input, d_gray, width, height, channels);
            cudaDeviceSynchronize();
        }
        
        unsigned char* d_blurred;
        cudaMalloc(&d_blurred, gray_size);
        
        GaussianBlurKernel<<<grid_size, block_size>>>(d_gray, d_blurred, width, height, blur_radius);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_gray, d_blurred, gray_size, cudaMemcpyDeviceToHost);
        
        std::string out_path = output_dir + "/" + filename + "_blur.png";
        SaveImage(out_path.c_str(), h_gray, width, height, 1);
        
        cudaFree(d_blurred);
        
        if (verbose) {
            std::cout << "  Saved blurred: " << out_path << "\n";
        }
    }
    
    if (mode == "edge" || mode == "all") {
        unsigned char* d_edges;
        cudaMalloc(&d_edges, gray_size);
        
        GrayscaleKernel<<<grid_size, block_size>>>(d_input, d_gray, width, height, channels);
        cudaDeviceSynchronize();
        
        SobelEdgeKernel<<<grid_size, block_size>>>(d_gray, d_edges, width, height);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_gray, d_edges, gray_size, cudaMemcpyDeviceToHost);
        
        std::string out_path = output_dir + "/" + filename + "_edge.png";
        SaveImage(out_path.c_str(), h_gray, width, height, 1);
        
        cudaFree(d_edges);
        
        if (verbose) {
            std::cout << "  Saved edges: " << out_path << "\n";
        }
    }
    
    if (mode == "enhance" || mode == "all") {
        EnhanceKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, channels, 1.2f, 10);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);
        
        std::string out_path = output_dir + "/" + filename + "_enhanced.png";
        SaveImage(out_path.c_str(), h_output, width, height, channels);
        
        if (verbose) {
            std::cout << "  Saved enhanced: " << out_path << "\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    delete[] h_gray;
    delete[] h_output;
    FreeImage(h_input);
    
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_output);
    cudaFree(d_temp);
    
    return elapsed;
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     CUDA Aerial Image Processing Pipeline                     ║\n";
    std::cout << "║     GPU-Accelerated Image Processing for Large Datasets       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    ProgramArgs args;
    if (!ParseArguments(argc, argv, args)) {
        return 1;
    }
    
    int device_count;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    
    if (cuda_status != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No CUDA-capable GPU detected!\n";
        std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_status) << "\n";
        return 1;
    }
    
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    
    std::cout << "GPU Device: " << device_prop.name << "\n";
    std::cout << "  Compute Capability: " << device_prop.major << "." << device_prop.minor << "\n";
    std::cout << "  Total Global Memory: " << (device_prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "  Multiprocessors: " << device_prop.multiProcessorCount << "\n\n";
    
    try {
        fs::create_directories(args.output_dir);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return 1;
    }
    
    std::vector<std::string> image_files = GetImageFiles(args.input_dir, args.num_images);
    
    if (image_files.empty()) {
        std::cerr << "Error: No image files found in " << args.input_dir << std::endl;
        return 1;
    }
    
    std::cout << "Processing Configuration:\n";
    std::cout << "  Input directory: " << args.input_dir << "\n";
    std::cout << "  Output directory: " << args.output_dir << "\n";
    std::cout << "  Processing mode: " << args.mode << "\n";
    std::cout << "  Images to process: " << image_files.size() << "\n";
    std::cout << "  Blur radius: " << args.blur_radius << "\n\n";
    
    std::cout << "Processing images...\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    double total_time = 0.0;
    int success_count = 0;
    int fail_count = 0;
    
    std::ofstream log_file(args.output_dir + "/processing_log.txt");
    log_file << "CUDA Image Processing Log\n";
    log_file << "=========================\n\n";
    log_file << "GPU: " << device_prop.name << "\n";
    log_file << "Mode: " << args.mode << "\n";
    log_file << "Blur Radius: " << args.blur_radius << "\n\n";
    
    auto pipeline_start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < image_files.size(); ++i) {
        std::string filename = fs::path(image_files[i]).filename().string();
        std::cout << "[" << (i + 1) << "/" << image_files.size() << "] " << filename;
        
        if (args.verbose) {
            std::cout << "\n";
        }
        
        double process_time = ProcessImage(image_files[i], args.output_dir, 
                                            args.mode, args.blur_radius, args.verbose);
        
        if (process_time >= 0) {
            total_time += process_time;
            success_count++;
            
            if (!args.verbose) {
                std::cout << " - " << process_time << " ms\n";
            } else {
                std::cout << "  Processing time: " << process_time << " ms\n";
            }
            
            log_file << filename << ": " << process_time << " ms (SUCCESS)\n";
        } else {
            fail_count++;
            std::cout << " - FAILED\n";
            log_file << filename << ": FAILED\n";
        }
    }
    
    auto pipeline_end = std::chrono::high_resolution_clock::now();
    double pipeline_time = std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start).count();
    
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    std::cout << "Processing Summary:\n";
    std::cout << "  Images processed: " << success_count << "/" << image_files.size() << "\n";
    std::cout << "  Failed: " << fail_count << "\n";
    std::cout << "  Total GPU processing time: " << total_time << " ms\n";
    std::cout << "  Total pipeline time: " << pipeline_time << " ms\n";
    std::cout << "  Average time per image: " << (total_time / success_count) << " ms\n";
    std::cout << "  Output saved to: " << args.output_dir << "\n\n";
    
    log_file << "\nSummary:\n";
    log_file << "  Images processed: " << success_count << "/" << image_files.size() << "\n";
    log_file << "  Failed: " << fail_count << "\n";
    log_file << "  Total GPU processing time: " << total_time << " ms\n";
    log_file << "  Total pipeline time: " << pipeline_time << " ms\n";
    log_file << "  Average time per image: " << (total_time / success_count) << " ms\n";
    log_file.close();
    
    std::cout << "Log file saved to: " << args.output_dir << "/processing_log.txt\n";
    std::cout << "\nDone!\n\n";
    
    return 0;
}

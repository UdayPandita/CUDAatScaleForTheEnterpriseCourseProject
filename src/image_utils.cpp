#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_utils.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>

static std::string GetExtension(const char* filename) {
    std::string name(filename);
    size_t pos = name.rfind('.');
    if (pos == std::string::npos) return "";
    
    std::string ext = name.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

unsigned char* LoadImage(const char* filename, int* width, int* height, int* channels) {
    int desired_channels = 0;
    unsigned char* data = stbi_load(filename, width, height, channels, desired_channels);
    if (data == nullptr) {
        return nullptr;
    }
    return data;
}

int SaveImage(const char* filename, const unsigned char* data, int width, int height, int channels) {
    std::string ext = GetExtension(filename);
    
    int result = 0;
    
    if (ext == ".png") {
        result = stbi_write_png(filename, width, height, channels, data, width * channels);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        result = stbi_write_jpg(filename, width, height, channels, data, 95);
    } else if (ext == ".bmp") {
        result = stbi_write_bmp(filename, width, height, channels, data);
    } else if (ext == ".tga") {
        result = stbi_write_tga(filename, width, height, channels, data);
    } else {
        std::string png_name = std::string(filename) + ".png";
        result = stbi_write_png(png_name.c_str(), width, height, channels, data, width * channels);
    }
    
    return result;
}

void FreeImage(unsigned char* data) {
    if (data != nullptr) {
        stbi_image_free(data);
    }
}

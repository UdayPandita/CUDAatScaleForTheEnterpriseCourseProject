#ifndef IMAGE_UTILS_H_
#define IMAGE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

unsigned char* LoadImage(const char* filename, int* width, int* height, int* channels);
int SaveImage(const char* filename, const unsigned char* data, int width, int height, int channels);
void FreeImage(unsigned char* data);

#ifdef __cplusplus
}
#endif

#endif

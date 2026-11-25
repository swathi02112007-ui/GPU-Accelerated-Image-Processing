
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                           \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(error) << '\n';   \
            return 1;                                         \
        }                                                     \
    } while (0)

struct Image {
    int width;
    int height;
    std::vector<unsigned char> pixels;
};

bool LoadImage(const char* filename, Image& image) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open input image.\n";
        return false;
    }

    std::string format;
    int maxValue;

    file >> format >> image.width >> image.height >> maxValue;
    file.get();

    if (format != "P6" || maxValue != 255 ||
        image.width <= 0 || image.height <= 0) {
        std::cerr << "Invalid PPM image.\n";
        return false;
    }

    size_t size = static_cast<size_t>(image.width) *
                  image.height * 3;

    image.pixels.resize(size);
    file.read(reinterpret_cast<char*>(image.pixels.data()), size);

    return file.good() || file.eof();
}

bool SaveImage(const char* filename, const Image& image) {
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot create output image.\n";
        return false;
    }

    file << "P6\n" << image.width << " "
         << image.height << "\n255\n";

    file.write(
        reinterpret_cast<const char*>(image.pixels.data()),
        image.pixels.size()
    );

    return file.good();
}

__host__ __device__
int GetGray(const unsigned char* pixels, int width,
            int x, int y) {
    int index = (y * width + x) * 3;

    int r = pixels[index];
    int g = pixels[index + 1];
    int b = pixels[index + 2];

    return (299 * r + 587 * g + 114 * b) / 1000;
}

__host__ __device__
unsigned char CalculateEdge(
    const unsigned char* pixels,
    int width, int height, int x, int y) {

    int left   = max(0, x - 1);
    int right  = min(width - 1, x + 1);
    int top    = max(0, y - 1);
    int bottom = min(height - 1, y + 1);

    int tl = GetGray(pixels, width, left, top);
    int tc = GetGray(pixels, width, x, top);
    int tr = GetGray(pixels, width, right, top);

    int ml = GetGray(pixels, width, left, y);
    int mr = GetGray(pixels, width, right, y);

    int bl = GetGray(pixels, width, left, bottom);
    int bc = GetGray(pixels, width, x, bottom);
    int br = GetGray(pixels, width, right, bottom);

    int gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
    int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;

    int magnitude = abs(gx) + abs(gy);

    return static_cast<unsigned char>(
        min(magnitude, 255)
    );
}

void SobelCPU(const Image& input, Image& output) {
    output.width = input.width;
    output.height = input.height;
    output.pixels.resize(input.pixels.size());

    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            unsigned char edge = CalculateEdge(
                input.pixels.data(),
                input.width, input.height, x, y
            );

            int index = (y * input.width + x) * 3;

            output.pixels[index] = edge;
            output.pixels[index + 1] = edge;
            output.pixels[index + 2] = edge;
        }
    }
}

__global__
void SobelKernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    unsigned char edge = CalculateEdge(
        input, width, height, x, y
    );

    int index = (y * width + x) * 3;

    output[index] = edge;
    output[index + 1] = edge;
    output[index + 2] = edge;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: ./sobel_edge input.ppm output.ppm\n";
        return 1;
    }

    Image input, cpuOutput, gpuOutput;

    if (!LoadImage(argv[1], input)) {
        return 1;
    }

    auto cpuStart = std::chrono::high_resolution_clock::now();

    SobelCPU(input, cpuOutput);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration<double, std::milli>(
        cpuEnd - cpuStart
    ).count();

    unsigned char *d_input, *d_output;

    size_t bytes = input.pixels.size();

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input, input.pixels.data(),
        bytes, cudaMemcpyHostToDevice
    ));

    dim3 block(16, 16);

    dim3 grid(
        (input.width + block.x - 1) / block.x,
        (input.height + block.y - 1) / block.y
    );

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    SobelKernel<<<grid, block>>>(
        d_input, d_output, input.width, input.height
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTime = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    gpuOutput.width = input.width;
    gpuOutput.height = input.height;
    gpuOutput.pixels.resize(bytes);

    CUDA_CHECK(cudaMemcpy(
        gpuOutput.pixels.data(), d_output,
        bytes, cudaMemcpyDeviceToHost
    ));

    bool valid = (cpuOutput.pixels == gpuOutput.pixels);

    if (!SaveImage(argv[2], gpuOutput)) {
        return 1;
    }

    std::cout << "\n========== SOBEL EDGE DETECTION RESULTS ==========\n";
    std::cout << "Image dimensions: "
              << input.width << " x " << input.height << '\n';
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "GPU kernel time: " << gpuTime << " ms\n";
    std::cout << "Output validation: "
              << (valid ? "PASSED" : "FAILED") << '\n';
    std::cout << "Output file: " << argv[2] << '\n';

    if (gpuTime > 0.0f) {
        std::cout << "Kernel speedup: "
                  << cpuTime / gpuTime << "x\n";
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return valid ? 0 : 1;
}

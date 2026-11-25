
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                      \
  do {                                                        \
    cudaError_t error = (call);                               \
    if (error != cudaSuccess) {                                \
      std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                << std::endl;                                 \
      return 1;                                                \
    }                                                         \
  } while (0)

struct Image {
  int width;
  int height;
  std::vector<unsigned char> pixels;
};

bool LoadImage(const std::string& filename, Image& image) {
  std::ifstream file(filename, std::ios::binary);

  if (!file) {
    std::cerr << "Cannot open image: " << filename << std::endl;
    return false;
  }

  std::string magic;
  int max_value;

  file >> magic >> image.width >> image.height >> max_value;

  if (magic != "P6" || max_value != 255 ||
      image.width <= 0 || image.height <= 0) {
    std::cerr << "Invalid PPM image." << std::endl;
    return false;
  }

  file.get();

  image.pixels.resize(
      static_cast<size_t>(image.width) * image.height * 3);

  file.read(reinterpret_cast<char*>(image.pixels.data()),
            image.pixels.size());

  return file.good() ||
         file.gcount() == static_cast<std::streamsize>(
                              image.pixels.size());
}

bool SaveImage(const std::string& filename, const Image& image,
               const std::vector<unsigned char>& pixels) {
  std::ofstream file(filename, std::ios::binary);

  if (!file) {
    return false;
  }

  file << "P6\n" << image.width << " " << image.height << "\n255\n";

  file.write(reinterpret_cast<const char*>(pixels.data()),
             pixels.size());

  return file.good();
}

void CpuGrayscale(const Image& input,
                  std::vector<unsigned char>& output) {
  const size_t pixel_count =
      static_cast<size_t>(input.width) * input.height;

  output.resize(pixel_count * 3);

  for (size_t i = 0; i < pixel_count; ++i) {
    const unsigned char r = input.pixels[i * 3];
    const unsigned char g = input.pixels[i * 3 + 1];
    const unsigned char b = input.pixels[i * 3 + 2];

    const unsigned char gray = static_cast<unsigned char>(
        (299 * static_cast<int>(r) +
         587 * static_cast<int>(g) +
         114 * static_cast<int>(b)) / 1000);

    output[i * 3] = gray;
    output[i * 3 + 1] = gray;
    output[i * 3 + 2] = gray;
  }
}

__global__ void GpuGrayscale(const unsigned char* input,
                             unsigned char* output,
                             int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const int index = (y * width + x) * 3;

  const unsigned char r = input[index];
  const unsigned char g = input[index + 1];
  const unsigned char b = input[index + 2];

  const unsigned char gray = static_cast<unsigned char>(
      (299 * static_cast<int>(r) +
         587 * static_cast<int>(g) +
         114 * static_cast<int>(b)) / 1000);

  output[index] = gray;
  output[index + 1] = gray;
  output[index + 2] = gray;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: ./grayscale <input.ppm> <output.ppm>\n";
    return 1;
  }

  Image image;

  if (!LoadImage(argv[1], image)) {
    return 1;
  }

  const size_t bytes = image.pixels.size();

  std::vector<unsigned char> cpu_output;
  std::vector<unsigned char> gpu_output(bytes);

  const auto cpu_start = std::chrono::high_resolution_clock::now();

  CpuGrayscale(image, cpu_output);

  const auto cpu_end = std::chrono::high_resolution_clock::now();

  const double cpu_ms =
      std::chrono::duration<double, std::milli>(
          cpu_end - cpu_start).count();

  unsigned char* d_input = nullptr;
  unsigned char* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  CUDA_CHECK(cudaMemcpy(d_input, image.pixels.data(),
                        bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((image.width + 15) / 16,
            (image.height + 15) / 16);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  GpuGrayscale<<<grid, block>>>(
      d_input, d_output, image.width, image.height);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float gpu_ms = 0.0f;

  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output,
                        bytes, cudaMemcpyDeviceToHost));

  const bool same_output = cpu_output == gpu_output;

  const std::string output_path = argv[2];

  if (!SaveImage(output_path, image, gpu_output)) {
    std::cerr << "Could not save output image.\n";
    return 1;
  }

  std::cout << "\n========== GRAYSCALE RESULTS ==========\n";
  std::cout << "Image dimensions: " << image.width
            << " x " << image.height << "\n";
  std::cout << "CPU time: " << cpu_ms << " ms\n";
  std::cout << "GPU kernel time: " << gpu_ms << " ms\n";
  std::cout << "Output validation: "
            << (same_output ? "PASSED" : "FAILED") << "\n";
  std::cout << "Output file: " << output_path << "\n";

  if (gpu_ms > 0.0f) {
    std::cout << "Kernel speedup: "
              << cpu_ms / gpu_ms << "x\n";
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);

  return same_output ? 0 : 1;
}

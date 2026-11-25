
#include <cuda_runtime.h>

#include <chrono>
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

const int GAUSSIAN_KERNEL[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

void CpuGaussianBlur(const Image& input,
                     std::vector<unsigned char>& output) {
  output.resize(input.pixels.size());

  for (int y = 0; y < input.height; ++y) {
    for (int x = 0; x < input.width; ++x) {
      for (int c = 0; c < 3; ++c) {
        int sum = 0;

        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int px = x + kx;
            int py = y + ky;

            if (px < 0) px = 0;
            if (py < 0) py = 0;
            if (px >= input.width) px = input.width - 1;
            if (py >= input.height) py = input.height - 1;

            int index = (py * input.width + px) * 3 + c;

            sum += input.pixels[index] *
                   GAUSSIAN_KERNEL[ky + 1][kx + 1];
          }
        }

        int output_index = (y * input.width + x) * 3 + c;
        output[output_index] =
            static_cast<unsigned char>(sum / 16);
      }
    }
  }
}

__global__ void GpuGaussianBlur(const unsigned char* input,
                                unsigned char* output,
                                int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const int weights[3][3] = {
      {1, 2, 1},
      {2, 4, 2},
      {1, 2, 1}
  };

  for (int c = 0; c < 3; ++c) {
    int sum = 0;

    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        int px = x + kx;
        int py = y + ky;

        if (px < 0) px = 0;
        if (py < 0) py = 0;
        if (px >= width) px = width - 1;
        if (py >= height) py = height - 1;

        int index = (py * width + px) * 3 + c;

        sum += input[index] * weights[ky + 1][kx + 1];
      }
    }

    int output_index = (y * width + x) * 3 + c;
    output[output_index] =
        static_cast<unsigned char>(sum / 16);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: ./gaussian_blur <input.ppm> <output.ppm>\n";
    return 1;
  }

  Image image;

  if (!LoadImage(argv[1], image)) {
    return 1;
  }

  const size_t bytes = image.pixels.size();

  std::vector<unsigned char> cpu_output;
  std::vector<unsigned char> gpu_output(bytes);

  auto cpu_start = std::chrono::high_resolution_clock::now();

  CpuGaussianBlur(image, cpu_output);

  auto cpu_end = std::chrono::high_resolution_clock::now();

  double cpu_ms =
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

  GpuGaussianBlur<<<grid, block>>>(
      d_input, d_output, image.width, image.height);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float gpu_ms = 0.0f;

  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output,
                        bytes, cudaMemcpyDeviceToHost));

  bool same_output = cpu_output == gpu_output;

  if (!SaveImage(argv[2], image, gpu_output)) {
    std::cerr << "Could not save output image.\n";
    return 1;
  }

  std::cout << "\n========== GAUSSIAN BLUR RESULTS ==========\n";
  std::cout << "Image dimensions: " << image.width
            << " x " << image.height << "\n";
  std::cout << "CPU time: " << cpu_ms << " ms\n";
  std::cout << "GPU kernel time: " << gpu_ms << " ms\n";
  std::cout << "Output validation: "
            << (same_output ? "PASSED" : "FAILED") << "\n";
  std::cout << "Output file: " << argv[2] << "\n";

  if (gpu_ms > 0.0f) {
    std::cout << "Kernel speedup: "
              << cpu_ms / gpu_ms << "x\n";
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return same_output ? 0 : 1;
}

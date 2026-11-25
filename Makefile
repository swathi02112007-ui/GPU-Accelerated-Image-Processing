
NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++17

TARGETS = grayscale gaussian_blur sobel_edge

all: $(TARGETS)

grayscale: src/grayscale.cu
	$(NVCC) $(NVCC_FLAGS) src/grayscale.cu -o grayscale

gaussian_blur: src/gaussian_blur.cu
	$(NVCC) $(NVCC_FLAGS) src/gaussian_blur.cu -o gaussian_blur

sobel_edge: src/sobel_edge.cu
	$(NVCC) $(NVCC_FLAGS) src/sobel_edge.cu -o sobel_edge

clean:
	rm -f $(TARGETS)

.PHONY: all clean

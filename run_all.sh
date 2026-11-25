#!/bin/bash
set -e

echo "======================================"
echo " CUDA IMAGE PROCESSING PROJECT"
echo "======================================"

# Create output directories
mkdir -p output/grayscale
mkdir -p output/blurred
mkdir -p output/edges

# Extract the 4096 image if it is not already available
if [ ! -f input/image_4096.ppm ] && [ -f input/image_4096.ppm.gz ]; then
    echo "Extracting 4096 x 4096 input image..."
    gzip -dk input/image_4096.ppm.gz
fi

# Process all four image resolutions
SIZES=(512 1024 2048 4096)

for SIZE in "${SIZES[@]}"; do

    echo ""
    echo "======================================"
    echo " Processing Image: ${SIZE} x ${SIZE}"
    echo "======================================"

    echo ""
    echo "----- GRAYSCALE CONVERSION -----"

    ./grayscale \
        "input/image_${SIZE}.ppm" \
        "output/grayscale/image_${SIZE}_gray.ppm"

    echo ""
    echo "----- GAUSSIAN BLUR -----"

    ./gaussian_blur \
        "input/image_${SIZE}.ppm" \
        "output/blurred/image_${SIZE}_blur.ppm"

    echo ""
    echo "----- SOBEL EDGE DETECTION -----"

    ./sobel_edge \
        "input/image_${SIZE}.ppm" \
        "output/edges/image_${SIZE}_edges.ppm"

    echo ""
    echo "Completed ${SIZE} x ${SIZE} image processing."
done

echo ""
echo "======================================"
echo " ALL IMAGE PROCESSING TASKS COMPLETED"
echo "======================================"

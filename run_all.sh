
#!/bin/bash

set -e

echo "======================================"
echo " CUDA IMAGE PROCESSING PROJECT"
echo "======================================"

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

done

echo ""
echo "======================================"
echo " ALL IMAGE PROCESSING TASKS COMPLETED"
echo "======================================"

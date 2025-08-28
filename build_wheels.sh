#!/bin/bash
set -e

# Define paths
KERNELS_DIR="qtexture/kernels"
METAL_FILE="$KERNELS_DIR/kernels.metal"
AIR_FILE="$KERNELS_DIR/kernels.air"
METALLIB_FILE="$KERNELS_DIR/kernels.metallib"

echo "Building kernels.metallib..."

# Ensure the metal file exists
if [ ! -f "$METAL_FILE" ]; then
    echo "Error: kernels.metal not found at $METAL_FILE"
    exit 1
fi

# Build the .air intermediate file
xcrun -sdk macosx metal -c "$METAL_FILE" -o "$AIR_FILE"

# Build the .metallib from the .air file
xcrun -sdk macosx metallib "$AIR_FILE" -o "$METALLIB_FILE"

# Clean up the intermediate .air file
rm "$AIR_FILE"

echo "Built kernels.metallib at $METALLIB_FILE"

# Create the wheel file using the standard python -m build command
python setup.py install

echo "Wheel file created in the 'dist' directory."
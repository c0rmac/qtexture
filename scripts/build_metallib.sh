#!/bin/bash
set -e
cd "$(dirname "$0")/../qtexture/kernels"
xcrun -sdk macosx metal -c kernels.metal -o kernels.air
xcrun -sdk macosx metallib kernels.air -o kernels.metallib
rm kernels.air
echo "Built kernels.metallib"

#!/bin/bash

# 检查必要的目录
if [ ! -d "/mnt/env/TensorRT-10.2.0.19" ]; then
    echo "Error: TensorRT not found at /mnt/env/TensorRT-10.2.0.19"
    exit 1
fi

if [ ! -d "/usr/local/cuda-12.4" ]; then
    echo "Error: CUDA not found at /usr/local/cuda-12.4"
    exit 1
fi

if [ ! -d "/ultralytics/c++/Submodule/TPL/av_opencv" ]; then
    echo "Error: OpenCV not found at /ultralytics/c++/Submodule/TPL/av_opencv"
    exit 1
fi

# 创建并进入构建目录
rm -rf build
mkdir -p build
cd build

# 配置 CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 \
    -DTENSORRT_ROOT=/mnt/env/TensorRT-10.2.0.19 \
    -DOpenCV_DIR=/ultralytics/c++/Submodule/TPL/av_opencv

# 编译
make -j$(nproc)

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "编译成功！"
else
    echo "编译失败！"
    exit 1
fi
cmake_minimum_required(VERSION 3.10)
project(trt_infer CUDA CXX)

# 设置 CUDA 策略
cmake_policy(SET CMP0146 NEW)  # 添加这行来使用新的 CUDA 查找策略

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Output/Lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Output/Lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Output/Lib)

# # 添加头文件搜索路径
# include_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/Detection
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/Pose
# )

# TensorRT
set(TENSORRT_ROOT /mnt/env/TensorRT-10.2.0.19)
if(NOT EXISTS ${TENSORRT_ROOT})
    message(FATAL_ERROR "TensorRT not found at ${TENSORRT_ROOT}")
endif()
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
if(NOT EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
    message(FATAL_ERROR "CUDA not found at ${CUDA_TOOLKIT_ROOT_DIR}")
endif()
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# 查找 CUDA 包
find_package(CUDAToolkit REQUIRED)  # 替换 find_package(CUDA REQUIRED)

# OpenCV
set(OpenCV_DIR /ultralytics/c++/Submodules/TPL/av_opencv)
if(NOT EXISTS ${OpenCV_DIR})
    message(FATAL_ERROR "OpenCV not found at ${OpenCV_DIR}")
endif()

include_directories(/ultralytics/c++/Include/Thirdpart/av_opencv)
link_directories(${OpenCV_DIR})

# 手动设置 OpenCV 库
set(OpenCV_LIBS
    ${OpenCV_DIR}/libopencv_core.so
    ${OpenCV_DIR}/libopencv_imgproc.so
    ${OpenCV_DIR}/libopencv_highgui.so
    ${OpenCV_DIR}/libopencv_imgcodecs.so
    ${OpenCV_DIR}/libopencv_calib3d.so
    ${OpenCV_DIR}/libopencv_dnn.so
    ${OpenCV_DIR}/libopencv_features2d.so
    ${OpenCV_DIR}/libopencv_flann.so
    ${OpenCV_DIR}/libopencv_ml.so
    ${OpenCV_DIR}/libopencv_objdetect.so
    ${OpenCV_DIR}/libopencv_photo.so
    ${OpenCV_DIR}/libopencv_stitching.so
    ${OpenCV_DIR}/libopencv_video.so
    ${OpenCV_DIR}/libopencv_videoio.so
)

# 添加源文件
add_executable(trt_infer
    Detection.cpp
    src/Detection/trt_infer.cpp

    # PoseEstimation.cpp
    # src/Pose/trt_infer.cpp
)

# 链接库
target_link_libraries(trt_infer PRIVATE
    ${TENSORRT_ROOT}/lib/libnvinfer.so
    ${TENSORRT_ROOT}/lib/libnvinfer_plugin.so
    ${CUDA_LIBRARIES}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
    ${OpenCV_LIBS}
)

# 设置编译选项
target_compile_options(trt_infer PRIVATE
    -Wall
    -Wextra
    -O3
    -Wno-deprecated-declarations  # 忽略废弃API的警告
    -Wno-unused-parameter        # 忽略未使用参数的警告
)

# 设置 CUDA 编译选项
set_target_properties(trt_infer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 设置运行时库路径
set_target_properties(trt_infer PROPERTIES
    INSTALL_RPATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64;${OpenCV_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
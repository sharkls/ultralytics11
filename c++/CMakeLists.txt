# 需要的CMake的最小版本号
cmake_minimum_required(VERSION 3.10)

# 执行文件另取别名
project(testAlgLib)

# 编译设置
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PLATFORM 86)

# 输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output/Lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output/Lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Output)

# CUDA配置
if(CMAKE_PLATFORM MATCHES "86")
    # CUDA环境和编译器设置
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    
    # CUDA路径设置
    set(CUDA_DIR /usr/local/cuda-12.4)
    set(CUDNN_DIR /usr)
    
    # CUDA编译选项
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")
    
    # 添加CUDA相关目录
    include_directories(${CUDA_DIR}/include ${CUDNN_DIR}/include)
    link_directories(${CUDA_DIR}/lib64 ${CUDNN_DIR}/lib/x86_64-linux-gnu)
endif()

add_compile_options(-shared)
add_definitions(-w)   # 不显示warning

message("CMAKE_SOURCE_DIR= " ${CMAKE_SOURCE_DIR})

# 包含头文件的路径(Fastdds 、 Proto定义文件)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/FastddsSer/data)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/FastddsSer/param)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ProtoSer/data)
include_directories(${CMAKE_SOURCE_DIR}/Submodules/ProtoSer/param)
# 包含头文件的路径
include_directories(${CMAKE_SOURCE_DIR}/Src)
include_directories(${CMAKE_SOURCE_DIR}/Include/AlgParam)
include_directories(${CMAKE_SOURCE_DIR}/Include/Interface)
include_directories(${CMAKE_SOURCE_DIR}/Include/Common)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/fastdds)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/av_opencv)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/yaml-cpp)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/tinyxml2)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/glog)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/protobuf)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/googletest)
include_directories(${CMAKE_SOURCE_DIR}/Include/Thirdpart/gflags)

# TensorRT 链接
set(TENSORRT_ROOT /mnt/env/TensorRT-10.2.0.19)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# 链接动态库的路径
link_directories(${CMAKE_SOURCE_DIR}/Output/Lib)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/fastdds)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/av_opencv)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/protobuf)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/googletest)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/gflags)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/yaml-cpp)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/tinyxml2)
link_directories(${CMAKE_SOURCE_DIR}/Submodules/TPL/glog)


# 子目录的路径
add_subdirectory(Src)
add_subdirectory(Submodules/FastddsSer)
add_subdirectory(Submodules/ProtoSer)

# 源文件的路径
aux_source_directory(. HW_SRC_FILES)

if(CMAKE_PLATFORM MATCHES "86")
    #生成可执行程序
    add_executable(${PROJECT_NAME} TestMultiModalDetection.cpp)
    # add_executable(${PROJECT_NAME} TestPoseEstimation.cpp)
    # add_executable(${PROJECT_NAME} TestObjectLocation.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_imgproc)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_imgcodecs)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_highgui)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_calib3d)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_dnn)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_features2d)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_flann)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_ml)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_objdetect)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_photo)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_stitching)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_video)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_videoio)
    target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml2)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
    target_link_libraries(${PROJECT_NAME} PRIVATE glog)
    target_link_libraries(${PROJECT_NAME} PRIVATE protobuf)
    target_link_libraries(${PROJECT_NAME} PRIVATE FastddsSer)
    target_link_libraries(${PROJECT_NAME} PRIVATE ProtoSer)
    target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin  nvonnxparser) # TensorRT nvparsers
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} PRIVATE cudart)


    # 链接各算法模块的动态库
    target_link_libraries(${PROJECT_NAME} PRIVATE PoseEstimationAlg)
    target_link_libraries(${PROJECT_NAME} PRIVATE ObjectLocationAlg)
    target_link_libraries(${PROJECT_NAME} PRIVATE MultiModalFusionAlg)
endif()
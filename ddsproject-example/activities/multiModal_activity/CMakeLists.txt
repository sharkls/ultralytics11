cmake_minimum_required(VERSION 3.22)

project(multiModal LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

set(multiModal_activity_PROTO_LIB multiModal_acticity_proto)
add_subdirectory(proto)

aux_source_directory(. SRC_FILES)

# CUDA配置
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

# 添加CUDA实现文件
set(SRC_FILES ${SRC_FILES} DepthConverter.cu)

#find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/include)
link_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")


pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWSCALE REQUIRED libswscale)
pkg_check_modules(AVRESAMPLE REQUIRED libavresample)

# 打印FFmpeg信息（可选）
message(STATUS "FFmpeg library status:")
message(STATUS "    avformat: ${AVFORMAT_VERSION}")
message(STATUS "    avcodec: ${AVCODEC_VERSION}")
message(STATUS "    avutil: ${AVUTIL_VERSION}")
message(STATUS "    swscale: ${SWSCALE_VERSION}")


# 对应方法一：生成activity的可执行程序
add_executable(multiModal_exec ${SRC_FILES})
target_link_libraries(multiModal_exec PRIVATE
    common
    activity_base
    node
    idl
    proto
    ${multiModal_activity_PROTO_LIB}
    ${ACTIVITY_IDL_LIB}
    )

target_link_libraries(multiModal_exec PRIVATE pthread)
target_link_libraries(multiModal_exec PRIVATE glog.so)

target_link_libraries(multiModal_exec PRIVATE 
    protobuf 
    absl_log_internal_check_op
    absl_log_internal_message)

target_link_libraries(multiModal_exec PRIVATE 
    fastrtps
    fastcdr)

target_link_libraries(multiModal_exec PRIVATE
    -lboost_iostreams
    )

target_link_libraries(multiModal_exec PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_highgui)

# 包含FFmpeg头文件目录
target_include_directories(multiModal_exec PRIVATE
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
)

# 链接FFmpeg库
target_link_libraries(multiModal_exec PRIVATE
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${AVRESAMPLE_LIBRARIES}
    #opencv_core
    #opencv_imgproc
    #opencv_imgcodecs
    #opencv_highgui
    pthread
    m
    ZLIB::ZLIB
)


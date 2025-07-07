cmake_minimum_required(VERSION 3.22)

project(VisualizationActivity LANGUAGES CXX)

set(VisualizationActivity_PROTO_LIB VisualizationActivity_proto)
add_subdirectory(proto)


find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)

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






link_directories(${CMAKE_SOURCE_DIR}/Output/Lib)


aux_source_directory(. SRC_FILES)






# 对应方法一：生成activity的可执行程序
add_library(push_lib pushStream.cpp)

# 包含FFmpeg头文件目录
target_include_directories(push_lib PRIVATE
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
)

message(${AVFORMAT_INCLUDE_DIRS})
message(${AVCODEC_INCLUDE_DIRS})
message(${AVUTIL_INCLUDE_DIRS})
message(${SWSCALE_INCLUDE_DIRS})

message(${AVFORMAT_LIBRARIES})
message(${AVCODEC_LIBRARIES})
message(${AVUTIL_LIBRARIES})
message(${SWSCALE_LIBRARIES})

# 链接FFmpeg库
target_link_libraries(push_lib PRIVATE
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${AVRESAMPLE_LIBRARIES}
    pthread
    m
    ZLIB::ZLIB
)



#add_executable(VisualizationActivity_exec ${SRC_FILES})
add_executable(VisualizationActivity_exec VisualizationActivity.cpp)

target_include_directories(VisualizationActivity_exec PRIVATE ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/include)
#target_link_directories(VisualizationActivity_exec PRIVATE ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib)

target_link_libraries(VisualizationActivity_exec PRIVATE
    common
    activity_base
    node
    idl
    proto
    ${VisualizationActivity_PROTO_LIB}
    ${ACTIVITY_IDL_LIB}
    # PoseEstimationAlg
    # VisualizationAlg
    push_lib
    )

target_link_libraries(VisualizationActivity_exec PRIVATE pthread)
target_link_libraries(VisualizationActivity_exec PRIVATE glog.so)

target_link_libraries(VisualizationActivity_exec PRIVATE 
    protobuf 
    absl_log_internal_check_op
    absl_log_internal_message)

target_link_libraries(VisualizationActivity_exec PRIVATE 
    fastrtps
    fastcdr)

# 添加OpenCV库
target_link_libraries(VisualizationActivity_exec PRIVATE
    ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib/libopencv_core.so
    ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib/libopencv_imgcodecs.so
    ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib/libopencv_imgproc.so
    ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib/libopencv_highgui.so
)


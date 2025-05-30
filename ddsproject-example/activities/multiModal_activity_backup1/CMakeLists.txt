cmake_minimum_required(VERSION 3.22)

project(multiModal LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

set(multiModal_activity_PROTO_LIB multiModal_acticity_proto)
add_subdirectory(proto)

aux_source_directory(. SRC_FILES)

# find_package(OpenCV REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/include)
link_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")


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
    # ${OpenCV_LIBS}
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

target_link_libraries(PoseEstimationActivity_exec PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_highgui)


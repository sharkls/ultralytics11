cmake_minimum_required(VERSION 3.22)

project(MultiModalFusionActivity LANGUAGES CXX)

set(MultiModalFusionActivity_PROTO_LIB MultiModalFusionActivity_proto)
add_subdirectory(proto)

aux_source_directory(. SRC_FILES)

find_package(OpenCV REQUIRED)

# 对应方法一：生成activity的可执行程序
add_executable(MultiModalFusionActivity ${SRC_FILES})
target_link_libraries(MultiModalFusionActivity PRIVATE
    common
    activity_base
    node
    idl
    proto
    ${MultiModalFusionActivity_PROTO_LIB}
    ${ACTIVITY_IDL_LIB}
    ${OpenCV_LIBS})

target_link_libraries(MultiModalFusionActivity PRIVATE pthread)
target_link_libraries(MultiModalFusionActivity PRIVATE glog.so)

target_link_libraries(MultiModalFusionActivity PRIVATE 
    protobuf 
    absl_log_internal_check_op
    absl_log_internal_message)

target_link_libraries(MultiModalFusionActivity PRIVATE 
    fastrtps
    fastcdr)


cmake_minimum_required(VERSION 3.22)

project(ObjectLocationActivity LANGUAGES CXX)

set(ObjectLocationActivity_PROTO_LIB ObjectLocationActivity_proto)
add_subdirectory(proto)

include_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/include)
link_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv/lib)
link_directories(${CMAKE_SOURCE_DIR}/Output/Lib)


aux_source_directory(. SRC_FILES)

# 对应方法一：生成activity的可执行程序
add_executable(ObjectLocationActivity_exec ${SRC_FILES})
target_link_libraries(ObjectLocationActivity_exec PRIVATE
    common
    activity_base
    node
    idl
    proto
    ${ObjectLocationActivity_PROTO_LIB}
    ${ACTIVITY_IDL_LIB}
    ObjectLocationAlg
    )

target_link_libraries(ObjectLocationActivity_exec PRIVATE pthread)
target_link_libraries(ObjectLocationActivity_exec PRIVATE glog.so)

target_link_libraries(ObjectLocationActivity_exec PRIVATE 
    protobuf 
    absl_log_internal_check_op
    absl_log_internal_message)

target_link_libraries(ObjectLocationActivity_exec PRIVATE 
    fastrtps
    fastcdr)

# 添加OpenCV库
target_link_libraries(ObjectLocationActivity_exec PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_highgui)

# 对应方法二：生成activity的动态库
# add_library(ObjectLocationActivity SHARED ${SRC_FILES})
# target_link_libraries(ObjectLocationActivity PRIVATE
#     common
#     activity_base
#     node
#     idl
#     proto
#     ${TEST0_ACTIVITY_PROTO_LIB}
#     ${ACTIVITY_IDL_LIB})

# target_link_libraries(ObjectLocationActivity PRIVATE pthread)
# target_link_libraries(ObjectLocationActivity PRIVATE glog.so)

# target_link_libraries(ObjectLocationActivity PRIVATE 
#     protobuf 
#     absl_log_internal_check_op
#     absl_log_internal_message)

# target_link_libraries(ObjectLocationActivity PRIVATE 
#     fastrtps
#     fastcdr)
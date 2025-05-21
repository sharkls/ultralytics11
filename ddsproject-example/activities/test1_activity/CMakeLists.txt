cmake_minimum_required(VERSION 3.22)

project(test1_activity LANGUAGES CXX)

set(TEST1_ACTIVITY_PROTO_LIB test1_activity_proto)
add_subdirectory(proto)

aux_source_directory(. SRC_FILES)

# 对应方法一：生成activity的可执行程序
add_executable(test1_activity_exec ${SRC_FILES})
target_link_libraries(test1_activity_exec PRIVATE
    common
    activity_base
    node
    idl
    proto
    ${TEST1_ACTIVITY_PROTO_LIB}
    ${ACTIVITY_IDL_LIB})

target_link_libraries(test1_activity_exec PRIVATE
    pthread)

target_link_libraries(test1_activity_exec PRIVATE glog.so)

target_link_libraries(test1_activity_exec PRIVATE 
    protobuf 
    absl_log_internal_check_op
    absl_log_internal_message)

target_link_libraries(test1_activity_exec PRIVATE 
    fastrtps
    fastcdr)

# 对应方法二：生成activity的动态库
# add_library(test1_activity SHARED ${SRC_FILES})
# target_link_libraries(test1_activity PRIVATE
#     common
#     activity_base
#     node
#     idl
#     proto
#     ${TEST1_ACTIVITY_PROTO_LIB}
#     ${ACTIVITY_IDL_LIB})

# target_link_libraries(test1_activity PRIVATE pthread)

# target_link_libraries(test1_activity PRIVATE glog.so)

# target_link_libraries(test1_activity PRIVATE 
#     protobuf 
#     absl_log_internal_check_op
#     absl_log_internal_message)

# target_link_libraries(test1_activity PRIVATE 
#     fastrtps
#     fastcdr)
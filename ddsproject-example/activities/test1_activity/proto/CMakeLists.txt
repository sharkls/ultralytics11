cmake_minimum_required(VERSION 3.22)

project(test1_activity_proto LANGUAGES CXX)

if (GENERATE_PROTO)
    if (NOT DEFINED PROTOBUF_PROTOC_EXECUTABLE)
        set(PROTOBUF_PROTOC_EXECUTABLE $ENV{PROTOBUF_PROTOC_EXECUTABLE})
    else()
        set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_PROTOC_EXECUTABLE})
    endif()
    message(${PROTOBUF_PROTOC_EXECUTABLE})
    set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/)
    set(PROTO_DST_DIR ${PROJECT_SOURCE_DIR}/)
    if (EXISTS ${PROTO_DST_DIR} AND IS_DIRECTORY ${PROTO_DST_DIR})
    else()
        file(MAKE_DIRECTORY ${PROTO_DST_DIR})
    endif()

    file(GLOB PROTO_FILES ${PROTO_SRC_DIR}/*.proto)

    list(APPEND PROTO_FLAGS -I${PROTO_SRC_DIR}/)

    foreach(msg ${PROTO_FILES})
        get_filename_component(FIL_WE ${msg} NAME_WE)
        list(APPEND ACTIVITY1_PROTO_SRCS ${PROTO_DST_DIR}/${FIL_WE}.pb.cc)
        list(APPEND ACTIVITY1_PROTO_HDRS ${PROTO_DST_DIR}/${FIL_WE}.pb.h)
        execute_process(
            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROTO_DST_DIR} ${msg}
        )
    endforeach()
    set_source_files_properties(${ACTIVITY1_PROTO_SRCS} ${ACTIVITY1_PROTO_HDRS} PROPERTIES GENERATED TRUE)

    add_library(${TEST1_ACTIVITY_PROTO_LIB} SHARED ${ACTIVITY1_PROTO_SRCS})
    target_link_libraries(${TEST1_ACTIVITY_PROTO_LIB} PRIVATE 
        protobuf 
        absl_log_internal_check_op
        absl_log_internal_message)

else ()
    set(TEST1_ACTIVITY_PROTO_SRC_FILES_DIR ${CMAKE_SOURCE_DIR}/activities/test1_activity/proto)
    aux_source_directory(${TEST1_ACTIVITY_PROTO_SRC_FILES_DIR}/ TEST1_ACTIVITY_PROTO_SRC_FILES)
    add_library(${TEST1_ACTIVITY_PROTO_LIB} SHARED ${TEST1_ACTIVITY_PROTO_SRC_FILES})
    target_link_libraries(${TEST1_ACTIVITY_PROTO_LIB} PRIVATE 
        protobuf 
        absl_log_internal_check_op
        absl_log_internal_message)
endif()

cmake_minimum_required(VERSION 3.22)

project(ObjectLocationActivity_proto LANGUAGES CXX)

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
        list(APPEND ObjectLocationActivity_PROTO_SRCS ${PROTO_DST_DIR}/${FIL_WE}.pb.cc)
        list(APPEND ObjectLocationActivity_PROTO_HDRS ${PROTO_DST_DIR}/${FIL_WE}.pb.h)
        execute_process(
            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROTO_DST_DIR} ${msg}
        )
    endforeach()
    set_source_files_properties(${ObjectLocationActivity_PROTO_SRCS} ${ObjectLocationActivity_PROTO_HDRS} PROPERTIES GENERATED TRUE)

    add_library(${ObjectLocationActivity_PROTO_LIB} SHARED ${ObjectLocationActivity_PROTO_SRCS})
    target_link_libraries(${ObjectLocationActivity_PROTO_LIB} PRIVATE 
        protobuf 
        absl_log_internal_check_op
        absl_log_internal_message)

else ()
    set(ObjectLocationActivity_PROTO_SRC_FILES_DIR ${CMAKE_SOURCE_DIR}/activities/ObjectLocationActivity/proto)
    aux_source_directory(${ObjectLocationActivity_PROTO_SRC_FILES_DIR}/ ObjectLocationActivity_PROTO_SRC_FILES)
    add_library(${ObjectLocationActivity_PROTO_LIB} SHARED ${ObjectLocationActivity_PROTO_SRC_FILES})
    target_link_libraries(${ObjectLocationActivity_PROTO_LIB} PRIVATE 
        protobuf 
        absl_log_internal_check_op
        absl_log_internal_message)
endif()



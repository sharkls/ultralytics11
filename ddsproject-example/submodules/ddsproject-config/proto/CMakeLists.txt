cmake_minimum_required(VERSION 3.22)

project(proto LANGUAGES CXX)

set(LIBRARY_OUTPUT_PATH ${LIB_PATH}/)

file(GLOB PROTO_FILES ${PROTO_SRC_DIR}/*.proto)

list(APPEND PROTO_FLAGS -I${PROTO_SRC_DIR}/)

foreach(msg ${PROTO_FILES})
    get_filename_component(FIL_WE ${msg} NAME_WE)
    list(APPEND PROTO_SRCS ${PROTO_DST_DIR}/${FIL_WE}.pb.cc)
    list(APPEND PROTO_HDRS ${PROTO_DST_DIR}/${FIL_WE}.pb.h)
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --cpp_out=${PROTO_DST_DIR} ${msg}
    )
endforeach()
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

add_library(${PROTO_LIB} SHARED ${PROTO_SRCS})
target_link_libraries(${PROTO_LIB} PRIVATE 
    protobuf 
    absl_log_internal_check_op
    absl_log_internal_message)
cmake_minimum_required(VERSION 3.22)

project(example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FastDDS_DIR ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/fastDDS)
set(Glog_DIR ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/glog)
set(GoogleTest_DIR ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/googletest)
set(Protobuf_DIR ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/protobuf)
set(gflags_DIR ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/gflags)
set(OpenCV_LIBS ${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/av_opencv)

include_directories(${FastDDS_DIR}/include)
include_directories(${Glog_DIR}/include)
include_directories(${GoogleTest_DIR}/include)
include_directories(${Protobuf_DIR}/include)
include_directories(${gflags_DIR}/include)
include_directories(${OpenCV_LIBS}/include)
include_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/tinyxml2/include)

link_directories(${FastDDS_DIR}/lib)
link_directories(${Glog_DIR}/lib)
link_directories(${GoogleTest_DIR}/lib)
link_directories(${Protobuf_DIR}/lib)
link_directories(${gflags_DIR}/lib)
include_directories(${OpenCV_LIBS}/lib)
include_directories(${CMAKE_SOURCE_DIR}/submodules/ddsproject-thirdparty/tinyxml2/lib)

set(OUTPUT_PATH ${CMAKE_SOURCE_DIR}/output)
set(BIN_PATH ${OUTPUT_PATH}/bin)
set(LIB_PATH ${OUTPUT_PATH}/lib)
set(LIBRARY_OUTPUT_PATH ${LIB_PATH})
set(EXECUTABLE_OUTPUT_PATH ${BIN_PATH}/)

# proto
set(PROTO_LIB proto)
if (NOT DEFINED GENERATE_PROTO)
    set(GENERATE_PROTO OFF)
endif()
if (GENERATE_PROTO)
    if (NOT DEFINED PROTOBUF_PROTOC_EXECUTABLE)
        set(PROTOBUF_PROTOC_EXECUTABLE $ENV{PROTOBUF_PROTOC_EXECUTABLE})
    else()
        set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_PROTOC_EXECUTABLE})
    endif()
    message(${PROTOBUF_PROTOC_EXECUTABLE})
    set(PROTO_SRC_DIR ${CMAKE_SOURCE_DIR}/submodules/ddsproject-config/proto)
    set(PROTO_DST_DIR ${CMAKE_SOURCE_DIR}/output/gen/proto/activity_cfg)
    if (EXISTS ${PROTO_DST_DIR} AND IS_DIRECTORY ${PROTO_DST_DIR})
    else()
        file(MAKE_DIRECTORY ${PROTO_DST_DIR})
    endif()
else ()
    message(STATUS "proto----------------------")
    set(PROTO_SRC_FILES_DIR ${CMAKE_SOURCE_DIR}/output/gen/proto/activity_cfg)
    aux_source_directory(${PROTO_SRC_FILES_DIR}/ PROTO_SRC_FILES)
    add_library(${PROTO_LIB} SHARED ${PROTO_SRC_FILES})
    target_link_libraries(${PROTO_LIB} PRIVATE 
        protobuf 
        absl_log_internal_check_op
        absl_log_internal_message)
endif()

# idl
set(IDL_LIB idl)
if (NOT DEFINED GENERATE_IDL)
    set(GENERATE_IDL OFF)
endif()
if (GENERATE_IDL)
    if (NOT DEFINED FASTDDSGEN_EXECUTABLE)
        set(FASTDDSGEN_EXECUTABLE $ENV{FASTDDSGEN_EXECUTABLE})
    else()
        set(FASTDDSGEN_EXECUTABLE ${FASTDDSGEN_EXECUTABLE})
    endif()
    set(IDL_SRC_DIR ${CMAKE_SOURCE_DIR}/submodules/ddsproject-config/idl)
    set(IDL_DST_DIR ${CMAKE_SOURCE_DIR}/output/gen/idl)
    if (EXISTS ${IDL_DST_DIR} AND IS_DIRECTORY ${IDL_DST_DIR})
    else()
        file(MAKE_DIRECTORY ${IDL_DST_DIR})
    endif()
else ()
    message(STATUS "idl----------------------")
    set(IDL_SRC_FILES_DIR ${CMAKE_SOURCE_DIR}/output/gen/idl)
    file(GLOB dir_list ${IDL_SRC_FILES_DIR}/*)
    foreach(dir ${dir_list})
        file(GLOB IDL_SRC_FILES ${dir}/*.cxx)
    endforeach()
    add_library(${IDL_LIB} SHARED ${IDL_SRC_FILES})
    target_link_libraries(${IDL_LIB} PRIVATE 
        fastrtps
        fastcdr
        tinyxml2
        )
endif()

include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/output/gen/)

link_directories(${CMAKE_SOURCE_DIR}/lib)
link_directories(${LIB_PATH})

if (GENERATE_PROTO AND GENERATE_IDL)
    add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/ddsproject-config)
endif()

add_subdirectory(activities)
# add_subdirectory(MultiModalFusionActivity)
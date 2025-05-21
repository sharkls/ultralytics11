cmake_minimum_required(VERSION 3.22)

project(idl LANGUAGES CXX)

set(LIBRARY_OUTPUT_PATH ${LIB_PATH}/)

file(GLOB IDL_FILES ${IDL_SRC_DIR}/*.idl)

foreach(msg ${IDL_FILES})
    get_filename_component(FIL_WE ${msg} NAME_WE)
    set(MSG_DIR ${IDL_DST_DIR}/${FIL_WE})
    if (EXISTS ${MSG_DIR} AND IS_DIRECTORY ${MSG_DIR})
    else()
        file(MAKE_DIRECTORY ${MSG_DIR})
    endif()
    list(APPEND IDL_SRCS ${MSG_DIR}/${FIL_WE}.cxx)
    list(APPEND IDL_SRCS ${MSG_DIR}/${FIL_WE}CdrAux.ipp)
    list(APPEND IDL_SRCS ${MSG_DIR}/${FIL_WE}PubSubTypes.cxx)
    list(APPEND IDL_HDRS ${MSG_DIR}/${FIL_WE}.h)
    list(APPEND IDL_HDRS ${MSG_DIR}/${FIL_WE}CdrAux.hpp)
    list(APPEND IDL_HDRS ${MSG_DIR}/${FIL_WE}PubSubTypes.h)
    execute_process(
        COMMAND ${FASTDDSGEN_EXECUTABLE} -d ${MSG_DIR} ${msg} -replace
    )
endforeach()
set_source_files_properties(${IDL_SRCS} ${IDL_HDRS} PROPERTIES GENERATED TRUE)

add_library(${IDL_LIB} SHARED ${IDL_HDRS} ${IDL_SRCS})

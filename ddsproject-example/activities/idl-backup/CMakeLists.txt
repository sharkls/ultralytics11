cmake_minimum_required(VERSION 3.22)

project(activity_idl LANGUAGES CXX)

if (GENERATE_IDL)
    if (NOT DEFINED FASTDDSGEN_EXECUTABLE)
        set(FASTDDSGEN_EXECUTABLE $ENV{FASTDDSGEN_EXECUTABLE})
    else()
        set(FASTDDSGEN_EXECUTABLE ${FASTDDSGEN_EXECUTABLE})
    endif()
    set(IDL_SRC_DIR ${PROJECT_SOURCE_DIR}/)
    set(IDL_DST_DIR ${PROJECT_SOURCE_DIR}/)
    if (EXISTS ${IDL_DST_DIR} AND IS_DIRECTORY ${IDL_DST_DIR})
    else()
        file(MAKE_DIRECTORY ${IDL_DST_DIR})
    endif()

    file(GLOB IDL_FILES ${IDL_SRC_DIR}/*.idl)

    foreach(msg ${IDL_FILES})
        get_filename_component(FIL_WE ${msg} NAME_WE)
        set(MSG_DIR ${IDL_DST_DIR}/${FIL_WE})
        if (EXISTS ${MSG_DIR} AND IS_DIRECTORY ${MSG_DIR})
        else()
            file(MAKE_DIRECTORY ${MSG_DIR})
        endif()
        list(APPEND ACTIVITY_IDL_SRCS ${MSG_DIR}/${FIL_WE}.cxx)
        list(APPEND ACTIVITY_IDL_SRCS ${MSG_DIR}/${FIL_WE}CdrAux.ipp)
        list(APPEND ACTIVITY_IDL_SRCS ${MSG_DIR}/${FIL_WE}PubSubTypes.cxx)
        list(APPEND ACTIVITY_IDL_HDRS ${MSG_DIR}/${FIL_WE}.h)
        list(APPEND ACTIVITY_IDL_HDRS ${MSG_DIR}/${FIL_WE}CdrAux.hpp)
        list(APPEND ACTIVITY_IDL_HDRS ${MSG_DIR}/${FIL_WE}PubSubTypes.h)
        execute_process(
            COMMAND ${FASTDDSGEN_EXECUTABLE} -d ${MSG_DIR} ${msg} -replace
        )
    endforeach()
    set_source_files_properties(${ACTIVITY_IDL_SRCS} ${ACTIVITY_IDL_HDRS} PROPERTIES GENERATED TRUE)

    add_library(${ACTIVITY_IDL_LIB} SHARED ${ACTIVITY_IDL_HDRS} ${ACTIVITY_IDL_SRCS})

else ()
    set(ACTIVITY_IDL_SRC_FILES_DIR ${PROJECT_SOURCE_DIR})
    file(GLOB dir_list ${ACTIVITY_IDL_SRC_FILES_DIR}/*)
    foreach(dir ${dir_list})
        if (IS_DIRECTORY ${dir})
            aux_source_directory(${dir} ACTIVITY_IDL_SRC_FILES)
        endif()
    endforeach()
    add_library(${ACTIVITY_IDL_LIB} SHARED ${ACTIVITY_IDL_SRC_FILES})
    target_link_libraries(${ACTIVITY_IDL_LIB} PRIVATE 
        fastrtps
        fastcdr
        tinyxml2)
endif()



set(SGBM_SOURCES
    SGBM.cpp
)

set(SGBM_HEADERS
    SGBM.h
)

add_library(SGBM STATIC ${SGBM_SOURCES} ${SGBM_HEADERS})

target_include_directories(SGBM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(OpenCV REQUIRED)
target_link_libraries(SGBM PUBLIC ${OpenCV_LIBS}) 
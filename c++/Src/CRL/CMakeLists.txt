set(CRL_SOURCES
    CRL.cpp
)

set(CRL_HEADERS
    CRL.h
)

add_library(CRL STATIC ${CRL_SOURCES} ${CRL_HEADERS})

target_include_directories(CRL PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 查找LibTorch
find_package(Torch REQUIRED)

# 链接库
target_link_libraries(CRL PUBLIC 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# 设置C++标准
target_compile_features(CRL PUBLIC cxx_std_17) 
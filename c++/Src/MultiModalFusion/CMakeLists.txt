project(MultiModalFusionAlg)

message("CMAKE_SOURCE_DIR= " ${CMAKE_SOURCE_DIR})
message("TestAlg CMAKE_PLATFORM " ${CMAKE_PLATFORM})

# 编译选项
add_compile_options(
    -fvisibility=hidden
    -fPIC
    -shared
    -fext-numeric-literals
)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/Include/AlgParam
    ${CMAKE_SOURCE_DIR}/Include/Common
    ${CMAKE_SOURCE_DIR}/Include/Interface
    ${CMAKE_SOURCE_DIR}/Include/Thirdpart/av_opencv
    ${CMAKE_SOURCE_DIR}/Include/Thirdpart/glog
    ${CMAKE_SOURCE_DIR}/Include/Thirdpart/fastdds
    ${CMAKE_SOURCE_DIR}/Src/Common
    ${CMAKE_SOURCE_DIR}/Src/MultiModalFusion
    ${CMAKE_SOURCE_DIR}/Src/MultiModalFusion/Preprocess
    ${CMAKE_SOURCE_DIR}/Src/MultiModalFusion/Postprocess
    ${CMAKE_SOURCE_DIR}/Src/MultiModalFusion/Inference
    /mnt/env/TensorRT-10.2.0.19/include
)

# 源文件
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/Common HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/MultiModalFusion HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/MultiModalFusion/Preprocess HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/MultiModalFusion/Postprocess HW_SRC_FILES)
aux_source_directory(${CMAKE_SOURCE_DIR}/Src/MultiModalFusion/Inference HW_SRC_FILES)
aux_source_directory(. HW_SRC_FILES)

# 链接库路径
link_directories(
    ${CMAKE_SOURCE_DIR}/Output/Lib
    ${CMAKE_SOURCE_DIR}/Submodules/TPL
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/fastdds
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/protobuf
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/av_opencv
    ${CMAKE_SOURCE_DIR}/Submodules/TPL/glog
    /mnt/env/TensorRT-10.2.0.19/lib
)

# 生成库
add_library(${PROJECT_NAME} SHARED ${HW_SRC_FILES} ExportMultiModalFusionAlgLib.cpp)

# 链接依赖库
target_link_libraries(${PROJECT_NAME} PRIVATE
    # OpenCV库
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_calib3d
    opencv_dnn
    opencv_features2d
    opencv_flann
    opencv_photo
    opencv_stitching
    opencv_video
    opencv_videoio
    opencv_ml
    opencv_objdetect
    
    # 其他依赖
    tinyxml2
    FastddsSer
    ProtoSer
    stdc++fs
    glog
    
    # TensorRT
    nvinfer
    nvinfer_plugin
    # nvparsers
    nvonnxparser
    cudart
)
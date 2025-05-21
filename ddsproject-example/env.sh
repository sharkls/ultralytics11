#! /bin/bash

# export PROTOBUF_PROTOC_EXECUTABLE=/usr/local/protobuf/bin/protoc
# export FASTDDSGEN_EXECUTABLE=/opt/Fast-RTPS-Gen/scripts/fastddsgen

# #!/bin/bash
# CUR_DIR=$(pwd)
# echo ${CUR_DIR}
# ENV_FILE=~/.bashrc

# # 先删除旧的 LD_LIBRARY_PATH 行
# sed -i '/export LD_LIBRARY_PATH=/d' $ENV_FILE

# # 再追加新的 LD_LIBRARY_PATH
# echo "export LD_LIBRARY_PATH=$CUR_DIR/Output/Lib:$CUR_DIR/Submodules/TPL:$CUR_DIR/Submodules/TPL/av_opencv:$CUR_DIR/Submodules/TPL/fastdds:$CUR_DIR/Submodules/TPL/protobuf:$CUR_DIR/Submodules/TPL/yaml-cpp:$CUR_DIR/Submodules/TPL/tinyxml2:$CUR_DIR/Submodules/TPL/glog:$CUR_DIR/Submodules/TPL/gflags:$CUR_DIR/Submodules/TPL/googletest" >> $ENV_FILE
# source $ENV_FILE
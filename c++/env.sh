#!/bin/bash
CUR_DIR=$(pwd)
echo ${CUR_DIR}
ENV_FILE=~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUR_DIR/Output/Lib:$CUR_DIR/Submodules/TPL:$CUR_DIR/Submodules/TPL/av_opencv:$CUR_DIR/Submodules/TPL/fastdds:$CUR_DIR/Submodules/TPL/protobuf/lib:$CUR_DIR/Submodules/TPL/yaml-cpp/lib:$CUR_DIR/Submodules/TPL/tinyxml2/lib:$CUR_DIR/Submodules/TPL/glog/lib:$CUR_DIR/Submodules/TPL/gflags/lib:$CUR_DIR/Submodules/TPL/googletest/lib" >> $ENV_FILE
source $ENV_FILE
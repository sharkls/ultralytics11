#!/bin/bash
CUR_DIR=$(pwd)
echo ${CUR_DIR}
ENV_FILE=~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUR_DIR/Output/Lib:$CUR_DIR/Submodules/TPL:$CUR_DIR/Submodules/TPL/av_opencv:$CUR_DIR/Submodules/TPL/fastdds" >> $ENV_FILE
source $ENV_FILE
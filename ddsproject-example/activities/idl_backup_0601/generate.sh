#! /bin/bash

output_path=$1
src_path=$2

for file in $(ls $src_path/ | grep ".idl")
do
    name=$(echo ${file} | awk -F'[.]' '{print $1}')
    dst_path=${output_path}/${name}
    echo ${name} ${file} ${dst_path}
    mkdir -p ${dst_path}
    cd ${src_path}
    fastddsgen -d ${dst_path}/ ${file} -replace
    echo "generate " ${file} " in " ${dst_path} 
done
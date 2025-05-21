#! /bin/bash

output_path=$1
src_path=$2

dst_path=${output_path}/
mkdir -p ${dst_path}

for file in $(ls $src_path/ | grep ".proto")
do
    protoc -I${src_path}/ --cpp_out=${dst_path}  $src_path/${file}
    echo "generate " ${file} " in " ${output_path}
done

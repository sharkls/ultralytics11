#! /bin/bash

dst_path=$1
src_path=$2

echo ${src_path} ${dst_path}
for file in $(ls $src_path/ | grep ".proto")
do
    protoc -I${src_path}/ --cpp_out=${dst_path}  $src_path/${file}
    echo "generate " ${file} " in " ${dst_path}
done

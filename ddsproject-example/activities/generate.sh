#! /bin/bash

current_path=$1

idl_dst_dir=${current_path}/idl
idl_src_dir=${current_path}/idl
sh ${current_path}/idl/generate.sh ${idl_dst_dir} ${idl_src_dir}

test0_proto_dst_dir=${current_path}/test0_activity/proto
test0_proto_src_dir=${current_path}/test0_activity/proto
sh ${current_path}/test0_activity/proto/generate.sh ${test0_proto_dst_dir} ${test0_proto_src_dir}

test1_proto_dst_dir=${current_path}/test1_activity/proto
test1_proto_src_dir=${current_path}/test1_activity/proto
sh ${current_path}/test1_activity/proto/generate.sh ${test1_proto_dst_dir} ${test1_proto_src_dir}

test2_proto_dst_dir=${current_path}/test2_activity/proto
test2_proto_src_dir=${current_path}/test2_activity/proto
sh ${current_path}/test2_activity/proto/generate.sh ${test2_proto_dst_dir} ${test2_proto_src_dir}
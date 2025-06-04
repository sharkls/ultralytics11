#! /bin/bash

project_path=$(pwd)

generate_output_path=${project_path}/output/gen
mkdir -p ${generate_output_path}

sh ${project_path}/submodules/ddsproject-config/idl/generate.sh ${generate_output_path}/idl/ ${project_path}/submodules/ddsproject-config/idl

sh ${project_path}/submodules/ddsproject-config/proto/generate.sh ${generate_output_path}/proto/ ${project_path}/submodules/ddsproject-config/proto

sh ${project_path}/submodules/ddsproject-config/MultiModalFusionActivity/proto/generate.sh ${generate_output_path}/proto/ ${project_path}/MultiModalFusionActivity/proto

sh ${project_path}/activities/generate.sh ${project_path}/activities
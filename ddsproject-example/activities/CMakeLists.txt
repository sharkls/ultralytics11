cmake_minimum_required(VERSION 3.22)

project(activities LANGUAGES CXX)

set(ACTIVITY_IDL_LIB activity_idl)

add_subdirectory(idl)
add_subdirectory(test0_activity)
# add_subdirectory(test1_activity)
# add_subdirectory(test2_activity)\
add_subdirectory(MultiModalFusionActivity)
add_subdirectory(PoseEstimationActivity)
add_subdirectory(ObjectLocationActivity)
add_subdirectory(VisualizationActivity)
add_subdirectory(multiModal_activity)
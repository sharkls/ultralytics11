cd /ultralytics/ddsproject-example/bin
./master -c /ultralytics/ddsproject-example/activities/conf/master_node.conf -l /ultralytics/ddsproject-example/activities/conf/activity_launch.conf >/dev/null 2>&1 &
sleep 5
cd /ultralytics/ddsproject-example/output/bin
./multiModal_exec >/dev/null 2>&1 &
sleep 2
./ObjectClassifyActivity_exec >/dev/null 2>&1 &
sleep 1
./ObjectDetectionActivity_exec >/dev/null 2>&1 &
sleep 1
./ObjectLocationActivity_exec >/dev/null 2>&1 &
sleep 1
./PoseEstimationV2Activity_exec >/dev/null 2>&1 &
sleep 1
./VisualizationActivity_exec >/dev/null 2>&1 &



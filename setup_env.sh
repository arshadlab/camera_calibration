# start Gazebo with plugins
gazebo -s libgazebo_ros_factory.so -s libgazebo_ros_init.so -s libgazebo_ros_state.so empty.world &
sleep 3

# spawn camera 
ros2 run gazebo_ros spawn_entity.py -file ./camera.urdf -entity camera1  -x 0.0 -y 0.0 -z 0.6 -unpause
sleep 1

#spawn checkerboard
ros2 run gazebo_ros spawn_entity.py -file ./checkerboard/checkerboard.sdf  -entity checkerboard  -x 0.0 -y 0.0 -z 0.4 -unpause
sleep 1

#run cameracalibrator
ros2 run camera_calibration cameracalibrator --no-service-check -p checkerboard --size 8x6 --square 0.02 --ros-args -r image:=/camera1/image_raw -p camera:=/camera1 &

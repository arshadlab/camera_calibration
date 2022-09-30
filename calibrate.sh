for x in 0.0 0.09987 -0.1111
do
  for y in 0.0 0.1421 -0.1404
  do
     for z in 0.1 0.3 0.35
     do
        for skew in 0.0 -0.2 0.2
        do
             # pause gazebo before calling set_entity_state service .  Apparently setting entity state doesn't work reliablity with gazebo in running mode
             ros2 service call /pause_physics 'std_srvs/srv/Empty' {""}
             ros2 service call /set_entity_state gazebo_msgs/SetEntityState "{state: { name: 'checkerboard', pose: {position: {x: $x, y: $y, z: $z}, orientation: {x: $skew, y: $skew, z: $skew}}, reference_frame: world}}"
             ros2 service call /unpause_physics 'std_srvs/srv/Empty' {""}
             sleep .8
         done
     done     
  done
done


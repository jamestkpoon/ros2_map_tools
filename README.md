# ros2_map_tools

Requirements:
- [PCL](https://pointclouds.org/downloads/)
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)  e.g. via ```sudo apt install libopencv-dev```
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) e.g. via ```sudo apt install libeigen3-dev```

```
colcon build --packages-select ros2_map_tools
```

## pcl_to_map
Squashes cloud files into 2D binary occupancy maps compatible with the [ROS2 map_server](https://index.ros.org/p/nav2_map_server).

Supported input formats: .pcd, .obj

### Params
- cloud: path to cloud file
- resolution: occupancy map resolution. Default: 0.05
- z_min: minimum point z-value for inclusion. Default: 0.0
- z_max: maximum point z-value for inclusion. Default: 1.8
- voxel_min_points: minimum point count per occupancy grid cell for occupancy. Default: 1
- yaw: yaw value for map origin in metadata. Default: 0.0
- tf_ext: optional extension override for transform metadata. Default: follows extension of "cloud" param

### Usage
1. Load config/pcl_to_map.rviz in RViz
1.
   ```
   ros2 run ros2_map_tools pcl_to_map
   ```
1. Iteratively transform the cloud until the occupancy map looks OK:
   - translate:
      ```
      ros2 topic pub --once /pcl_to_map/translate geometry_msgs/msg/Vector3 "{x: 0.0, y: 0.0, z: 0.0 }"
      ```
    - rotate:
      ```
      ros2 topic pub --once /pcl_to_map/rotate geometry_msgs/msg/Vector3 "{x: 0.0, y: 0.0, z: 0.0 }"
      ```
1. To save:
   ```
   ros2 service call /pcl_to_map/save std_srvs/srv/Trigger
   ```
   This saves a .png and .yaml pair that can be loaded with map_server (see below), as well as an additional \_tf.yaml containing a transform from map frame to the input cloud (ie. an inverse of the final cumulative transformation).
1. To verify (optional):
   1. Enable the "Map" display in RViz
   1. 
      ```
      ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/path/to/map.yaml
      ```
      Note: you may need to also run
      ```
      ros2 run nav2_util lifecycle_bringup map_server
      ```

## dense_map_builder
Builds dense maps from timestamp-synchronized nav_msgs/msg/Odometry and sensor_msgs/msg/PointCloud2.

```
ros2 run ros2_map_tools dense_map_builder
```
Params:
- cloud_hz: optional cloud cache throttling frequency if > 0.0 . Default: -1.0
- trajectory: optional trajectory file in [TUM format](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
- cloud: outbound .pcd map path. Default: "cloud"
- voxel_size: voxel grid size for downsampling. Utilized iff > 0.0 . Default: -1.0
- voxel_min_points: minimum points per voxel for occupancy. Default: 1
- downsample_local_clouds: apply voxel downsampling to local point cloud msgs as they are received. Default: true if voxel_size > 0.0

To save the dense cloud once all your data has been published:
```
ros2 service call /dense_map_builder/save std_srvs/srv/Trigger
```
Note: This will clear the message buffers, unless there are no point clouds available.

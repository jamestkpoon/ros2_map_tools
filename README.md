# ros2_map_tools

Requirements:
- [PCL](https://pointclouds.org/downloads/) e.g. via ```sudo apt install libpcl-dev ros-${ROS_DISTRO}-pcl-ros```
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)  e.g. via ```sudo apt install libopencv-dev```
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) e.g. via ```sudo apt install libeigen3-dev```
- [Octomap](https://github.com/octomap/octomap)
- python3 modules: [cv2](https://pypi.org/project/opencv-python/), [numpy](https://numpy.org/install/), [scipy](https://scipy.org/install/), [yaml](https://pypi.org/project/PyYAML/)

```
colcon build --packages-select ros2_map_tools --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## pcl_to_map
Squashes 3D data files into 2D binary occupancy maps compatible with the [ROS2 map_server](https://index.ros.org/p/nav2_map_server).

Supported input formats: .pcd, .obj, .ot, .xyz

### Params
- file: input filepath
- occupancy_likelihood_threshold: probability threshold for occupancy (applicable formats: .ot). Default: 0.5
- resolution: occupancy map resolution. Default: 0.05
- z_min: minimum point z-value for inclusion. Default: 0.0
- z_max: maximum point z-value for inclusion. Default: 1.8
- voxel_min_points: minimum point count per occupancy grid cell for occupancy. Default: 1
- transform: optional comma-separated floating point initial transform as a 6-tuple (tx,ty,tz,rx,ry,rz) or 7-tuple (tx,ty,tz,qx,qy,qz,qw)
- yaw: yaw value for map origin in metadata. Default: 0.0
- tf_ext: optional extension override for transform metadata. Default: follows extension of "cloud" param

### Usage
1. In one terminal, load config/pcl_to_map.rviz in RViz, e.g.:
   ```
   rviz2 -d config/pcl_to_map.rviz
   ```
1. In a second terminal: 
   ```
   ros2 run ros2_map_tools pcl_to_map --ros-args ...
   ```
1. In a third terminal, iteratively transform the cloud until the occupancy map looks OK:
   - translate:
      ```
      ros2 topic pub --once /pcl_to_map/translate geometry_msgs/msg/Vector3 "{x: 0.0, y: 0.0, z: 0.0 }"
      ```
    - rotate:
      ```
      ros2 topic pub --once /pcl_to_map/rotate geometry_msgs/msg/Vector3 "{x: 0.0, y: 0.0, z: 0.0 }"
      ```
1. To save (suggested to call from third terminal):
   ```
   ros2 service call /pcl_to_map/save std_srvs/srv/Trigger
   ```
   This saves a .png and .yaml pair that can be loaded with map_server (see below). The .yaml also contains the final cumulative transform from map frame to the raw input.
1. To verify (optional):
   1. Enable the "Map" display in RViz
   1. From some other terminal not running RViz:
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
- odom_topic: optional odometry topic. If omitted, incoming cloud msgs with stamps beyond the range in the trajectory file will be rejected
- voxel_size: voxel grid size. Default: 0.05
- voxel_min_points: minimum points per voxel for downsampling inbound point clouds. Default: 1
- tree: outbound binary .ot path. Default: "tree"

To save the dense cloud from another terminal once all your data has been published:
```
ros2 service call /dense_map_builder/save std_srvs/srv/Trigger
```
Note: This will clear the message buffers, unless there are no point clouds available.

## map_aligner_ui.py
A "UI" to align >=2 2D maps together, to generate new map metadatas and a final combined map.
Future work may be towards automating this, given the assumption of maps that fit together cleanly and are suitable for alignment via feature matching.

```
ros2 run ros2_map_tools map_aligner_ui.py
```

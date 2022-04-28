#include <iostream>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2/buffer_core.h>
#include <tf2_ros/buffer_interface.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "ros2_map_tools/string_tools.hpp"

using namespace pcl;

#define AUTHORITY "default_authority"
#define TARGET_FRAME_OVERRIDE "target_frame"
#define SOURCE_FRAME_OVERRIDE "source_frame"

struct OctocloudStamped
{
    tf2::TimePoint stamp;
    std::shared_ptr<octomap::Pointcloud> cloud;
};


geometry_msgs::msg::TransformStamped from_tum(const std::string& line, const char sep = ' ')
{
    std::vector<double> doubles = split_string_to_doubles(line, sep);

    geometry_msgs::msg::TransformStamped stf;

    stf.header.stamp.sec = doubles[0];
    stf.header.stamp.nanosec = (doubles[0] - stf.header.stamp.sec) * 1e9;

    stf.transform.translation.x = doubles[1];
    stf.transform.translation.y = doubles[2];
    stf.transform.translation.z = doubles[3];
    stf.transform.rotation.x = doubles[4];
    stf.transform.rotation.y = doubles[5];
    stf.transform.rotation.z = doubles[6];
    stf.transform.rotation.w = doubles[7];

    return stf;
}

class DenseMapBuilder : public rclcpp::Node
{
    public:
        DenseMapBuilder() : Node("dense_map_builder")
        {
            poses_ = new std::vector<geometry_msgs::msg::TransformStamped>();
            auto trajectory_fp = declare_parameter<std::string>("trajectory", "");
            if(trajectory_fp != "") {
                std::ifstream file (trajectory_fp);
                if(file.is_open()) {
                    std::string line;
                    while(std::getline(file, line)) {
                        poses_->push_back(from_tum(line));
                    }
                }
                file.close();

                std::cout << "Loaded " << poses_->size() << " poses" << std::endl;
            }

            clouds_ = new std::vector<OctocloudStamped>();
            cloud_throttle_period_ns_ = 1e9 / declare_parameter<double>("cloud_hz", -1.0);

            float voxel_size = declare_parameter<double>("voxel_size", 0.05);
            voxelgrid_filter_ = new VoxelGrid<PointXYZ>();
            voxelgrid_filter_->setLeafSize (voxel_size, voxel_size, voxel_size);
            voxelgrid_filter_->setMinimumPointsNumberPerVoxel(declare_parameter<int>("voxel_min_points", 1));

            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                "odom", 10, std::bind(&DenseMapBuilder::odom_callback, this, std::placeholders::_1));
            cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "cloud", 10, std::bind(&DenseMapBuilder::cloud_callback, this, std::placeholders::_1));

            map_save_server_ = create_service<std_srvs::srv::Trigger>("~/save",
                std::bind(&DenseMapBuilder::save_svc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        }

    private:
        void odom_callback(const nav_msgs::msg::Odometry & msg)
        {
            geometry_msgs::msg::TransformStamped stf;
            stf.header = msg.header;
            stf.transform.translation.x = msg.pose.pose.position.x;
            stf.transform.translation.y = msg.pose.pose.position.y;
            stf.transform.translation.z = msg.pose.pose.position.z;
            stf.transform.rotation.x = msg.pose.pose.orientation.x;
            stf.transform.rotation.y = msg.pose.pose.orientation.y;
            stf.transform.rotation.z = msg.pose.pose.orientation.z;
            stf.transform.rotation.w = msg.pose.pose.orientation.w;

            poses_->push_back(stf);
        }

        void cloud_callback(const sensor_msgs::msg::PointCloud2& msg)
        {
            if(clouds_->empty() || (cloud_throttle_period_ns_ <= 0.0)) {
                cache_cloud(msg);
            } else {
                auto dt = tf2_ros::fromMsg(msg.header.stamp) - clouds_->back().stamp;
                if(dt.count() >= cloud_throttle_period_ns_) { cache_cloud(msg); }
            }
        }

        void cache_cloud(const sensor_msgs::msg::PointCloud2& msg)
        {
            OctocloudStamped cs;
            cs.stamp = tf2_ros::fromMsg(msg.header.stamp);

            // downsample
            PCLPointCloud2 pcl_pc2; pcl_conversions::toPCL(msg, pcl_pc2);
            PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
            fromPCLPointCloud2(pcl_pc2, *cloud);
            voxelgrid_filter_->setInputCloud (cloud);
            PointCloud<PointXYZ>::Ptr cloud_downsample(new PointCloud<PointXYZ>);
            voxelgrid_filter_->filter (*cloud_downsample);

            // octotree pointcloud
            cs.cloud = std::shared_ptr<octomap::Pointcloud>(new octomap::Pointcloud);
            for(const auto& pt : *cloud_downsample) {
                cs.cloud->push_back(octomap::point3d(pt.x, pt.y, pt.z));
            }

            clouds_->push_back(cs);
        }

        void save_svc(
            const std::shared_ptr<rmw_request_id_t> req_header,
            const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
            std::shared_ptr<std_srvs::srv::Trigger::Response> res)
        {
            (void)req_header; (void)req; // suppress unused

            if(clouds_->empty()) {
                res->message = "No clouds available";
                res->success = false;
                return;
            }

            // populate buffercore for lookups
            auto clouds_dur = clouds_->back().stamp - clouds_->front().stamp;
            tf2::BufferCore buffercore(clouds_dur + tf2::BUFFER_CORE_DEFAULT_CACHE_TIME);
            for(auto& stf : *poses_) {
                stf.header.frame_id = TARGET_FRAME_OVERRIDE;
                stf.child_frame_id = SOURCE_FRAME_OVERRIDE;
                buffercore.setTransform(stf, AUTHORITY);
            }
            poses_->clear();

            // populate octomap tree
            std::cout << "Populating tree ..." << std::endl;
            octomap::OcTree tree(voxelgrid_filter_->getLeafSize().x());
            int num_clouds = clouds_->size(), num_clouds_ok = 0, num_clouds_attempted = 0;
            for(const auto& cs : *clouds_) {
                ++num_clouds_attempted;
                if(buffercore.canTransform(TARGET_FRAME_OVERRIDE, SOURCE_FRAME_OVERRIDE, cs.stamp)) {
                    ++num_clouds_ok;
                    auto tf = buffercore.lookupTransform(TARGET_FRAME_OVERRIDE, SOURCE_FRAME_OVERRIDE, cs.stamp).transform;
                    octomap::pose6d pose(octomath::Vector3(tf.translation.x, tf.translation.y, tf.translation.z),
                        octomath::Quaternion(tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z));

                    tree.insertPointCloud(*cs.cloud, octomap::point3d(0,0,0), pose);

                    std::cout << "  " << num_clouds_attempted << " / " << num_clouds << std::endl;

                    if(!rclcpp::ok()) {
                        res->message = "Abort";
                        res->success = false;
                        return;
                    }
                }
            }
            clouds_->clear();
            std::cout << "Tree populated" << std::endl;

            // save
            if(tree.size() > 1) {
                tree.writeBinary(declare_parameter<std::string>("tree", "tree") + ".ot");

                res->message = std::to_string(tree.getNumLeafNodes()) + " leaf nodes from "
                    + std::to_string(num_clouds_ok) + "/" + std::to_string(num_clouds) + " clouds";
                res->success = true;
            } else {
                res->message = "No clouds utilized";
                res->success = false;
            }
        }

        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_server_;

        std::vector<geometry_msgs::msg::TransformStamped> *poses_;
        std::vector<OctocloudStamped> *clouds_;
        int64_t cloud_throttle_period_ns_;
        VoxelGrid<PointXYZ> *voxelgrid_filter_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto dmb = new DenseMapBuilder();
    rclcpp::spin(dmb->get_node_base_interface());

    rclcpp::shutdown();
    return 0;
}
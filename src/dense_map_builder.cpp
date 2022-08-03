#include <iostream>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2/buffer_core.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_eigen/tf2_eigen.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/string.hpp>

#include "ros2_map_tools/string_tools.hpp"

using namespace pcl;

#define AUTHORITY "default_authority"
#define TARGET_FRAME_OVERRIDE "target_frame"
#define SOURCE_FRAME_OVERRIDE "source_frame"

struct CloudPtrStamped
{
    tf2::TimePoint stamp;
    PointCloud<PointXYZRGB>::Ptr cloud;
};

geometry_msgs::msg::TransformStamped from_tum(const std::string &line, const char sep = ' ')
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
        load_trajectory(declare_parameter<std::string>("trajectory", ""));

        clouds_ = new std::vector<CloudPtrStamped>();
        cloud_throttle_period_ns_ = 1e9 / declare_parameter<double>("cloud_hz", -1.0);

        voxelgrid_filter_ = new VoxelGrid<PointXYZRGB>();
        const float voxel_size = declare_parameter<double>("voxel_size", 0.05);
        voxelgrid_filter_->setLeafSize(voxel_size, voxel_size, voxel_size);
        voxelgrid_filter_->setMinimumPointsNumberPerVoxel(declare_parameter<int>("voxel_min_points", 1));

        prob_thresh_ = declare_parameter<double>("occupancy_likelihood_threshold", 0.5);
        out_fp_ = declare_parameter<std::string>("out_path", "cloud") + ".ply";

        auto odom_topic = declare_parameter<std::string>("odom_topic", "");
        if (odom_topic != "")
        {
            odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                odom_topic, 10, std::bind(&DenseMapBuilder::odom_callback, this, std::placeholders::_1));
        }
        else
        {
            odom_sub_ = nullptr;
        }

        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "cloud", 10, std::bind(&DenseMapBuilder::cloud_callback, this, std::placeholders::_1));

        map_save_server_ = this->create_service<std_srvs::srv::Trigger>(
            "~/save",
            std::bind(&DenseMapBuilder::save_svc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

        map_save_sub_ = this->create_subscription<std_msgs::msg::String>(
            "~/load_trajectory_and_save", 10, std::bind(&DenseMapBuilder::save_cb, this, std::placeholders::_1));
    }

private:
    void odom_callback(const nav_msgs::msg::Odometry &msg)
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

    void cloud_callback(const sensor_msgs::msg::PointCloud2 &msg)
    {
        if (clouds_->empty() || (cloud_throttle_period_ns_ <= 0.0))
        {
            cache_cloud(msg);
        }
        else
        {
            auto dt = tf2_ros::fromMsg(msg.header.stamp) - clouds_->back().stamp;
            if (dt.count() >= cloud_throttle_period_ns_)
            {
                cache_cloud(msg);
            }
        }
    }

    void cache_cloud(const sensor_msgs::msg::PointCloud2 &msg)
    {
        CloudPtrStamped cs;
        cs.stamp = tf2_ros::fromMsg(msg.header.stamp);

        if (!odom_sub_ && !poses_->empty())
        {
            if (cs.stamp < tf2_ros::fromMsg(poses_->front().header.stamp))
            {
                RCLCPP_WARN(get_logger(), "Cloud rejected - timestamp before trajectory start");
                return;
            }
            if (cs.stamp > tf2_ros::fromMsg(poses_->back().header.stamp))
            {
                RCLCPP_WARN(get_logger(), "Cloud rejected  - timestamp after trajectory end");
                return;
            }
        }

        // downsample
        PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
        fromROSMsg(msg, *cloud);
        voxelgrid_filter_->setInputCloud(cloud);
        PointCloud<PointXYZRGB>::Ptr cloud_downsample(new PointCloud<PointXYZRGB>);
        voxelgrid_filter_->filter(*cloud_downsample);
        cs.cloud = cloud_downsample;

        clouds_->push_back(cs);
    }

    void save_svc(
        const std::shared_ptr<rmw_request_id_t> req_header,
        const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        (void)req_header;
        (void)req; // suppress unused

        if (clouds_->empty())
        {
            res->message = "No clouds available";
            res->success = false;
            return;
        }

        // populate buffercore for lookups
        auto clouds_dur = clouds_->back().stamp - clouds_->front().stamp;
        tf2::BufferCore buffercore(clouds_dur + tf2::BUFFER_CORE_DEFAULT_CACHE_TIME);
        for (auto &stf : *poses_)
        {
            stf.header.frame_id = TARGET_FRAME_OVERRIDE;
            stf.child_frame_id = SOURCE_FRAME_OVERRIDE;
            buffercore.setTransform(stf, AUTHORITY);
        }
        poses_->clear();

        // populate octomap tree
        RCLCPP_INFO(get_logger(), "Populating tree ...");
        octomap::OcTree tree(voxelgrid_filter_->getLeafSize().x());
        PointCloud<PointXYZRGB>::Ptr cloud_agg(new PointCloud<PointXYZRGB>);
        int num_clouds = clouds_->size(), num_clouds_ok = 0, num_clouds_attempted = 0;
        for (const auto &cs : *clouds_)
        {
            ++num_clouds_attempted;
            if (buffercore.canTransform(TARGET_FRAME_OVERRIDE, SOURCE_FRAME_OVERRIDE, cs.stamp))
            {
                ++num_clouds_ok;

                auto tf = buffercore.lookupTransform(TARGET_FRAME_OVERRIDE, SOURCE_FRAME_OVERRIDE, cs.stamp).transform;
                geometry_msgs::msg::Pose pose;
                pose.position.x = tf.translation.x;
                pose.position.y = tf.translation.y;
                pose.position.z = tf.translation.z;
                pose.orientation.x = tf.rotation.x;
                pose.orientation.y = tf.rotation.y;
                pose.orientation.z = tf.rotation.z;
                pose.orientation.w = tf.rotation.w;
                Eigen::Affine3d aff;
                tf2::fromMsg(pose, aff);

                PointCloud<PointXYZRGB> cloud_transformed;
                transformPointCloud(*cs.cloud, cloud_transformed, aff);
                *cloud_agg += cloud_transformed;
                octomap::Pointcloud octocloud;
                for (const auto &pt : cloud_transformed)
                {
                    octocloud.push_back(octomap::point3d(pt.x, pt.y, pt.z));
                }
                tree.insertPointCloud(octocloud, octomap::point3d(0, 0, 0));
                RCLCPP_INFO(get_logger(), (std::to_string(num_clouds_attempted) + " / " + std::to_string(num_clouds)).c_str());

                if (!rclcpp::ok())
                {
                    res->message = "Abort";
                    res->success = false;
                    return;
                }
            }
        }
        clouds_->clear();

        // save
        if (tree.size() > 1)
        {
            RCLCPP_INFO(get_logger(), "Assigning colors ...");

            PointCloud<PointXYZRGB> leaves;
            KdTreeFLANN<pcl::PointXYZRGB> kdtree;
            kdtree.setInputCloud(cloud_agg);
            float sqdl = voxelgrid_filter_->getLeafSize().x() * voxelgrid_filter_->getLeafSize().x();
            for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it)
            {
                if (it->getOccupancy() >= prob_thresh_)
                {
                    PointXYZRGB pt;
                    pt.x = it.getX();
                    pt.y = it.getY();
                    pt.z = it.getZ();

                    std::vector<int> pointIdxKNNSearch(1);
                    std::vector<float> pointKNNSquaredDistance(1);
                    if ((kdtree.nearestKSearch(pt, 1, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) && (pointKNNSquaredDistance[0] <= sqdl))
                    {
                        leaves.push_back((*cloud_agg)[pointIdxKNNSearch[0]]);
                    }
                }
            }

            io::savePLYFileBinary(out_fp_, leaves);
            res->message = std::to_string(leaves.size()) + " points from " + std::to_string(num_clouds_ok) + "/" + std::to_string(num_clouds) + " clouds";
            res->success = true;
            RCLCPP_INFO(get_logger(), res->message.c_str());
        }
        else
        {
            res->message = "No clouds utilized";
            res->success = false;
            RCLCPP_ERROR(get_logger(), res->message.c_str());
        }
    }

    void save_cb(const std_msgs::msg::String &msg)
    {
        load_trajectory(msg.data);

        auto header = std::make_shared<rmw_request_id_t>();
        auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto res = std::make_shared<std_srvs::srv::Trigger::Response>();
        save_svc(header, req, res);

        if (res->success)
        {
            RCLCPP_INFO(get_logger(), res->message.c_str());
        }
        else
        {
            RCLCPP_ERROR(get_logger(), res->message.c_str());
        }
    }

    bool load_trajectory(const std::string &trajectory_fp)
    {
        if (trajectory_fp == "")
        {
            return false;
        }

        unsigned int previous_size = poses_->size();
        std::ifstream file(trajectory_fp);
        if (file.is_open())
        {
            std::string line;
            while (std::getline(file, line))
            {
                poses_->push_back(from_tum(line));
            }
        }
        file.close();

        unsigned int lines_added = poses_->size() - previous_size;
        RCLCPP_INFO(get_logger(), ("Loaded " + std::to_string(lines_added) + " poses").c_str());

        return lines_added;
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_server_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr map_save_sub_;

    std::vector<geometry_msgs::msg::TransformStamped> *poses_;
    std::vector<CloudPtrStamped> *clouds_;
    int64_t cloud_throttle_period_ns_;
    VoxelGrid<PointXYZRGB> *voxelgrid_filter_;
    double prob_thresh_;
    std::string out_fp_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto dmb = new DenseMapBuilder();
    rclcpp::spin(dmb->get_node_base_interface());

    rclcpp::shutdown();
    return 0;
}

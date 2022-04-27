#include <iostream>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <tf2/buffer_core.h>
#include <tf2_ros/buffer_interface.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/trigger.hpp>

using namespace pcl;
using namespace Eigen;

#define AUTHORITY "default_authority"
#define TARGET_FRAME_OVERRIDE "target_frame"
#define SOURCE_FRAME_OVERRIDE "source_frame"

struct CloudXYZPtrStamped
{
    tf2::TimePoint stamp;
    PointCloud<PointXYZ>::Ptr cloud;
};


std::vector<std::string> split_string(const std::string& s, const char sep)
{
    std::vector<std::string> out_;
    std::string substring_ = "";
    for(const char& c : s + sep)
    {
        if(c != sep) substring_ += c;
        else if(!substring_.empty())
        {
            out_.push_back(substring_);
            substring_.clear();
        }
    }

    return out_;
}

geometry_msgs::msg::TransformStamped from_tum(const std::string& line, const char sep = ' ')
{
    std::vector<double> doubles;
    for(const auto& s : split_string(line, sep)) {
        doubles.push_back(std::atof(s.c_str()));
    }

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
            }

            clouds_ = new std::vector<CloudXYZPtrStamped>();
            cloud_throttle_period_ns_ = 1e9 / declare_parameter<double>("cloud_hz", -1.0);

            float voxel_size = declare_parameter<double>("voxel_size", -1.0);
            if(voxel_size > 0.0) {
                voxelgrid_filter_ = new VoxelGrid<PointXYZ>();
                voxelgrid_filter_->setLeafSize (voxel_size, voxel_size, voxel_size);
                voxelgrid_filter_->setMinimumPointsNumberPerVoxel(declare_parameter<int>("voxel_min_points", 1));
                downsample_local_clouds_ = declare_parameter<bool>("downsample_local_clouds", true);
            } else {
                voxelgrid_filter_ = nullptr;
                downsample_local_clouds_ = false;
            }

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
            CloudXYZPtrStamped cs;
            cs.stamp = tf2_ros::fromMsg(msg.header.stamp);
            cs.cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);
            PCLPointCloud2 pcl_pc2; pcl_conversions::toPCL(msg, pcl_pc2);
            fromPCLPointCloud2(pcl_pc2, *cs.cloud);

            if(downsample_local_clouds_) {
                voxelgrid_filter_->setInputCloud (cs.cloud);
                PointCloud<PointXYZ>::Ptr cloud_downsample(new PointCloud<PointXYZ>);
                voxelgrid_filter_->filter (*cloud_downsample);
                cs.cloud = cloud_downsample;
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

            // aggregate clouds
            PointCloud<PointXYZ>::Ptr cloud_agg(new PointCloud<PointXYZ>);
            int num_ok_clouds = 0, num_clouds = clouds_->size();
            for(auto& cs : *clouds_) {
                if(buffercore.canTransform(TARGET_FRAME_OVERRIDE, SOURCE_FRAME_OVERRIDE, cs.stamp)) {
                    // apply affine transform
                    Affine3d transform = Affine3d::Identity();
                    auto tf = buffercore.lookupTransform(TARGET_FRAME_OVERRIDE, SOURCE_FRAME_OVERRIDE, cs.stamp).transform;
                    transform.translate(Vector3d(tf.translation.x, tf.translation.y, tf.translation.z));
                    transform.rotate(Quaterniond(tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z));
                    PointCloud<PointXYZ>::Ptr transformed_cloud (new PointCloud<PointXYZ> ());
                    transformPointCloud (*cs.cloud, *transformed_cloud, transform);

                    // append points to aggregate cloud, increment counter
                    *cloud_agg += *transformed_cloud;
                    ++num_ok_clouds;
                }

                cs.cloud->clear(); // free up a bit of memory during aggregation
            }
            clouds_->clear();

            // save
            if(!cloud_agg->empty()) {
                auto fp = declare_parameter<std::string>("cloud", "cloud") + ".pcd";
                if(voxelgrid_filter_) {
                    voxelgrid_filter_->setInputCloud (cloud_agg);
                    PointCloud<PointXYZ>::Ptr cloud_downsample(new PointCloud<PointXYZ>);
                    voxelgrid_filter_->filter (*cloud_downsample);
                    cloud_agg = cloud_downsample;
                }
                io::savePCDFile(fp, *cloud_agg);

                res->message = std::to_string(cloud_agg->size()) + " points from "
                    + std::to_string(num_ok_clouds) + "/" + std::to_string(num_clouds) + " clouds";
                res->success = true;
            } else {
                res->message = "No points aggregated";
                res->success = false;
            }
        }

        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_server_;

        std::vector<geometry_msgs::msg::TransformStamped> *poses_;
        std::vector<CloudXYZPtrStamped> *clouds_;
        int64_t cloud_throttle_period_ns_;
        VoxelGrid<PointXYZ> *voxelgrid_filter_;
        bool downsample_local_clouds_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto dmb = new DenseMapBuilder();
    rclcpp::spin(dmb->get_node_base_interface());

    rclcpp::shutdown();
    return 0;
}
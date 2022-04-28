#include <iostream>
#include <fstream>
#include <filesystem>

#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <Eigen/Dense>
#include <opencv2/highgui.hpp>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "halodi_ros2_pcl_tools/string_tools.hpp"

using namespace pcl;
using namespace Eigen;

#define OCCUPIED_CELL_VALUE 100
#define FREE_CELL_VALUE 0
#define OCCUPIED_PIXEL_VALUE 0
#define FREE_PIXEL_VALUE 255

Quaterniond from_euler(const double x, const double y, const double z)
{
    AngleAxisd xaa(x, Vector3d::UnitX());
    AngleAxisd yaa(y, Vector3d::UnitY());
    AngleAxisd zaa(z, Vector3d::UnitZ());

    return xaa * yaa * zaa;
}

class PclToMap : public rclcpp::Node
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PclToMap() : Node("pcl_to_map")
        {
            if(!load_cloud()) { return; }

            auto sub_qos = rclcpp::QoS(10); sub_qos.reliable(); sub_qos.durability_volatile();
            t_sub_ = create_subscription<geometry_msgs::msg::Vector3>("~/translate", sub_qos, std::bind(&PclToMap::t_cb, this, std::placeholders::_1));
            r_sub_ = create_subscription<geometry_msgs::msg::Vector3>("~/rotate",    sub_qos, std::bind(&PclToMap::r_cb, this, std::placeholders::_1));

            auto pub_qos = rclcpp::QoS(1); pub_qos.reliable(); pub_qos.transient_local();
            cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/cloud", pub_qos);
            og_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("~/map", pub_qos);

            map_save_server_ = create_service<std_srvs::srv::Trigger>("~/save",
                std::bind(&PclToMap::save_svc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

            og_msg_ptr_ = new nav_msgs::msg::OccupancyGrid();
            og_msg_ptr_->header.frame_id = "map";
            og_msg_ptr_->info.resolution = declare_parameter<double>("resolution", 0.05);

            z_filter_ = new PassThrough<PointXYZ>();
            z_filter_->setFilterFieldName ("z");
            z_filter_->setFilterLimits(
                declare_parameter<double>("zmin", 0.0),
                declare_parameter<double>("zmax", 1.8));
            voxelgrid_filter_ = new VoxelGrid<PointXYZ>();
            voxelgrid_filter_->setLeafSize(og_msg_ptr_->info.resolution, og_msg_ptr_->info.resolution, FLT_MAX);
            voxelgrid_filter_->setMinimumPointsNumberPerVoxel(declare_parameter<int>("voxel_min_points", 1));

            cumulative_transform_.setIdentity();
            update(transform_from_param());
        }

        bool cloud_loaded()
        {
            return ((cloud_ptr_ != nullptr) && !cloud_ptr_->empty());
        }

    private:
        void t_cb(const geometry_msgs::msg::Vector3::SharedPtr msg)
        {
            Affine3d transform = Affine3d::Identity();
            transform.translate(Vector3d(msg->x, msg->y, msg->z));
            update(transform);
        }

        void r_cb(const geometry_msgs::msg::Vector3::SharedPtr msg)
        {
            Affine3d transform = Affine3d::Identity();
            transform.rotate(from_euler(msg->x, msg->y, msg->z));
            update(transform);
        }

        void save_svc(
            const std::shared_ptr<rmw_request_id_t> req_header,
            const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
            std::shared_ptr<std_srvs::srv::Trigger::Response> res)
        {
            (void)req_header; (void)req; // suppress unused

            // write 2D map + metadata
            cv::Mat om_mat = cv::Mat(cv::Size(og_msg_ptr_->info.width, og_msg_ptr_->info.height), CV_8U, &og_msg_ptr_->data[0]),
                om_mat_flip = cv::Mat(om_mat.size(), CV_8U), occ_mask = cv::Mat(om_mat.size(), CV_8U);
            cv::flip(om_mat, om_mat_flip, 0);
            cv::inRange(om_mat_flip, OCCUPIED_CELL_VALUE, OCCUPIED_CELL_VALUE, occ_mask);
            om_mat_flip.setTo(OCCUPIED_PIXEL_VALUE, occ_mask);
            om_mat_flip.setTo(FREE_PIXEL_VALUE, 255 - occ_mask);
            std::string image_filename = cloud_in_stem_ + ".png";
            cv::imwrite(image_filename, om_mat_flip);

            std::ofstream yaml_stream;
            const auto yaw = declare_parameter<double>("yaw", 0.0);
            yaml_stream.open(cloud_in_stem_ + ".yaml");
            yaml_stream << "image: " << image_filename << "\n";
            yaml_stream << "resolution: " << og_msg_ptr_->info.resolution << "\n";
            yaml_stream << "origin: [ " << og_msg_ptr_->info.origin.position.x << ", " << og_msg_ptr_->info.origin.position.y << ", " << yaw << " ]" << "\n";
            yaml_stream << "occupied_thresh: 0.65" << "\n";
            yaml_stream << "free_thresh: 0.19" << "\n";
            yaml_stream << "negate: 0";
            yaml_stream.close();

            // write transform
            const auto ext = declare_parameter<std::string>("tf_ext", cloud_in_ext_);
            auto ct_inv = cumulative_transform_.inverse();
            auto translation = ct_inv.translation();
            Quaterniond rotation(ct_inv.rotation());
            yaml_stream.open(cloud_in_stem_ + "_tf.yaml");
            yaml_stream << "map: " << cloud_in_stem_ << ext << "\n";
            yaml_stream << "transform:\n";
            yaml_stream << "  translation:\n";
            yaml_stream << "    x: " << std::to_string(translation.x()) << "\n";
            yaml_stream << "    y: " << std::to_string(translation.y()) << "\n";
            yaml_stream << "    z: " << std::to_string(translation.z()) << "\n";
            yaml_stream << "  rotation:\n";
            yaml_stream << "    x: " << std::to_string(rotation.x()) << "\n";
            yaml_stream << "    y: " << std::to_string(rotation.y()) << "\n";
            yaml_stream << "    z: " << std::to_string(rotation.z()) << "\n";
            yaml_stream << "    w: " << std::to_string(rotation.w()) << "\n";
            yaml_stream.close();

            res->success = true;
        }

        bool load_cloud()
        {
            cloud_ptr_ = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);            

            const auto fp = declare_parameter<std::string>("file", "");
            auto prob_thresh = declare_parameter<double>("occupancy_likelihood_threshold", 0.5);
            if(fp != "") {
                auto path = std::filesystem::path(fp);
                cloud_in_stem_ = path.stem();
                cloud_in_ext_ = path.extension();
                if(cloud_in_ext_ == ".pcd") {
                    io::loadPCDFile<PointXYZ> (fp, *cloud_ptr_);
                } else if(cloud_in_ext_ == ".obj") {
                    TextureMesh::Ptr mesh(new TextureMesh);
                    if(OBJReader().read(fp, *mesh) == 0) {
                        fromPCLPointCloud2(mesh->cloud, *cloud_ptr_);
                    }
                }
                else if(cloud_in_ext_ == ".ot") {
                    octomap::OcTree tree(fp);
                    for(auto it=tree.begin_leafs(), end=tree.end_leafs(); it!= end; ++it) {
                        if(it->getOccupancy() >= prob_thresh) {
                            auto center = it.getCoordinate();
                            cloud_ptr_->push_back(PointXYZ(center.x(), center.y(), center.z()));
                        }
                    }
                }

                if(cloud_loaded()) {
                    std::cout << "Loaded " << fp << " (" << cloud_ptr_->size() << " points)" << std::endl;
                } else {
                    std::cout << "Unable to load " << fp << std::endl;
                }
            }

            return cloud_loaded();
        }

        void update(const Affine3d& transform)
        {
            // apply transform
            PointCloud<PointXYZ>::Ptr cloud_tf(new PointCloud<PointXYZ>);
            transformPointCloud (*cloud_ptr_, *cloud_tf, transform);
            cloud_ptr_ = cloud_tf;

            // crop along z axis, downsample to occupancy grid resolution
            z_filter_->setInputCloud(cloud_ptr_);
            PointCloud<PointXYZ>::Ptr cloud_zfilterered(new PointCloud<PointXYZ>);
            z_filter_->filter(*cloud_zfilterered);
            voxelgrid_filter_->setInputCloud(cloud_zfilterered);
            PointCloud<PointXYZ>::Ptr cloud_downsample(new PointCloud<PointXYZ>);
            voxelgrid_filter_->filter(*cloud_downsample);

            // resize occupancy grid
            MomentOfInertiaEstimation<PointXYZ> feature_extractor;
            feature_extractor.setInputCloud(cloud_downsample); feature_extractor.compute();
            PointXYZ min_point_AABB, max_point_AABB;
            feature_extractor.getAABB(min_point_AABB, max_point_AABB);

            og_msg_ptr_->info.width  = uint32_t(std::ceil((max_point_AABB.x - min_point_AABB.x) / og_msg_ptr_->info.resolution));
            og_msg_ptr_->info.height = uint32_t(std::ceil((max_point_AABB.y - min_point_AABB.y) / og_msg_ptr_->info.resolution));
            og_msg_ptr_->data.resize(og_msg_ptr_->info.height * og_msg_ptr_->info.width);
            og_msg_ptr_->info.origin.position.x = min_point_AABB.x;
            og_msg_ptr_->info.origin.position.y = min_point_AABB.y;

            // re-compute occupancy grid
            og_msg_ptr_->data.resize(og_msg_ptr_->info.width  * og_msg_ptr_->info.height);
            std::fill(og_msg_ptr_->data.begin(), og_msg_ptr_->data.end(), FREE_CELL_VALUE);
            for(const auto& pt : *cloud_downsample) {
                uint32_t r = std::round((pt.x - og_msg_ptr_->info.origin.position.x) / og_msg_ptr_->info.resolution),
                    c = std::round((pt.y - og_msg_ptr_->info.origin.position.y) / og_msg_ptr_->info.resolution);

                if((r < og_msg_ptr_->info.width) && (c < og_msg_ptr_->info.height)) {
                    uint32_t i = c * og_msg_ptr_->info.width + r;
                    og_msg_ptr_->data[i] = OCCUPIED_CELL_VALUE;
                }
            }

            // publish
            og_msg_ptr_->header.stamp = get_clock()->now();
            og_pub_->publish(*og_msg_ptr_);

            sensor_msgs::msg::PointCloud2 cloud_msg;
            toROSMsg(*cloud_ptr_, cloud_msg);
            cloud_msg.header = og_msg_ptr_->header;
            cloud_pub_->publish(cloud_msg);

            // update cumulative transform
            cumulative_transform_ = transform * cumulative_transform_;
        }

        Affine3d transform_from_param()
        {
            Affine3d transform = Affine3d::Identity();

            auto initial_transform = declare_parameter<std::string>("transform", "");
            auto doubles = split_string_to_doubles(initial_transform, ',');

            switch(doubles.size()) {
                case 6:
                    transform.translate(Vector3d(doubles[0], doubles[1], doubles[2]));
                    transform.rotate(from_euler(doubles[3], doubles[4], doubles[5]));
                    break;
                case 7:
                    transform.translate(Vector3d(doubles[0], doubles[1], doubles[2]));
                    transform.rotate(Quaterniond(doubles[6], doubles[3], doubles[4], doubles[5]));
                    break;
                default:
                    std::string err = "Invalid transform parameter: " + initial_transform;
                    RCLCPP_ERROR(get_logger(), err.c_str());
                    break;
            }

            return transform;
        }

        rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr t_sub_, r_sub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr og_pub_;
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_server_;
        nav_msgs::msg::OccupancyGrid* og_msg_ptr_;

        PointCloud<PointXYZ>::Ptr cloud_ptr_;
        std::string cloud_in_stem_, cloud_in_ext_;
        Affine3d cumulative_transform_;
        PassThrough<PointXYZ> *z_filter_;
        VoxelGrid<PointXYZ> *voxelgrid_filter_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto ptm = new PclToMap();
    bool ok = ptm->cloud_loaded();

    if(ok) { rclcpp::spin(ptm->get_node_base_interface()); }

    rclcpp::shutdown();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#ifndef ROS2_MAP_TOOLS_MAP_HPP
#define ROS2_MAP_TOOLS_MAP_HPP

#include <iostream>
#include <string>
#include <filesystem>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#define MAP_FREE_COLOR 255
#define MAP_OCC_COLOR 0


Eigen::Affine3d from_xy_yaw(const double x, const double y, const double yaw)
{
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translate(Eigen::Vector3d(x, y, 0.0));
    transform.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

    return transform;
}

Eigen::Affine3d from_xy_yaw(const Eigen::Vector3d& p)
{
    return from_xy_yaw(p[0], p[1], p[2]);
}

Eigen::Vector3d to_xy_yaw(const Eigen::Affine3d& transform)
{
    auto translation = transform.translation();
    auto euler = transform.rotation().eulerAngles(0, 1, 2);

    return Eigen::Vector3d(translation.x(), translation.y(), euler[2]);
}

Eigen::Affine3d from_yaml_tf(const YAML::Node& node)
{
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translate(
        Eigen::Vector3d(
            node["translation"]["x"].as<double>(),
            node["translation"]["y"].as<double>(),
            node["translation"]["z"].as<double>()
        )
    );
    transform.rotate(
        Eigen::Quaterniond(
            node["rotation"]["x"].as<double>(),
            node["rotation"]["y"].as<double>(),
            node["rotation"]["z"].as<double>(),
            node["rotation"]["w"].as<double>()
        )
    );

    return transform;
}

YAML::Node from_eigen(const Eigen::Affine3d& transform)
{
    YAML::Node node, translation, rotation;

    auto trans = transform.translation();
    translation["x"] = trans.x();
    translation["y"] = trans.y();
    translation["z"] = trans.z();

    auto quat = Eigen::Quaterniond(transform.rotation());
    rotation["x"] = quat.x();
    rotation["y"] = quat.y();
    rotation["z"] = quat.z();
    rotation["w"] = quat.w();

    node["translation"] = translation;
    node["rotation"] = rotation;

    return node;
}

void write_yaml_node(const YAML::Node& node, const std::filesystem::path& path)
{
    std::ofstream writer; writer.open(path);
    writer << node << "\n";
    writer.close();
}

void get_rotation_matrix_and_bounded_size(const double ccw_rads, const cv::Mat& image,
    cv::Mat& rot, cv::Size2f& size)
{
    // rotation matrix around image center
    cv::Point2f center_px(image.cols / 2, image.rows / 2);
    double angle = ccw_rads * 180 / M_PI;
    rot = cv::getRotationMatrix2D(center_px, angle, 1.0);

    // get output size and adjust translation
    size = cv::RotatedRect(center_px, image.size(), angle).boundingRect2f().size();
    rot.at<double>(0,2) += (size.width  - image.cols) / 2;
    rot.at<double>(1,2) += (size.height - image.rows) / 2;
}


class Map
{
    public:
        struct Masks
        {
            cv::Mat free, occupied, unknown;

            void set_unknown_from_free_and_occupied()
            {
                unknown = ~(free | occupied);
            }
        };

        struct PixelAndPose
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            cv::Point2f pt;
            Eigen::Vector3d pos;
        };

        static void write(const YAML::Node& _node, const cv::Mat& image, const std::filesystem::path& image_path)
        {
            auto node = YAML::Clone(_node);
            node["image"] = image_path.stem().string() + image_path.extension().string();

            cv::imwrite(image_path, image);

            auto yaml_path = image_path.parent_path() / (image_path.stem().string() + ".yaml");
            write_yaml_node(node, yaml_path);
        }

        Map(const std::string& yaml_fp, const double target_resolution=-1.0)
        {
            path_ = std::filesystem::path(yaml_fp);
            yaml_node_ = YAML::LoadFile(yaml_fp);

            auto image_fp = path_.parent_path() / yaml_node_["image"].as<std::string>();
            image_ = new cv::Mat(cv::imread(image_fp, cv::IMREAD_GRAYSCALE));

            auto origin_vector = yaml_node_["origin"].as<std::vector<double>>();
            origin_ = Eigen::Vector3d(origin_vector[0], origin_vector[1], origin_vector[2]);

            res_ = yaml_node_["resolution"].as<double>();

            double free_threshf = yaml_node_["free_thresh"].as<double>(),
                occ_threshf = yaml_node_["occupied_thresh"].as<double>(),
                unknown_pixel_valuef = (free_threshf + occ_threshf) / 2;
            free_thresh_ = std::round(255 * free_threshf);
            occ_thresh_ = std::round(255 * occ_threshf);
            unknown_pixel_value_ = std::round(255 * unknown_pixel_valuef);

            if((target_resolution > 0.0) && (target_resolution != res_)) {
                auto image_scaling_factor_ = res_ / target_resolution;
                res_ = target_resolution;
                cv::Size dsize(
                    std::round(image_->cols * image_scaling_factor_),
                    std::round(image_->rows * image_scaling_factor_)
                );

                cv::Mat image_resized;
                cv::resize(*image_, image_resized, dsize, cv::INTER_NEAREST);
                *image_ = image_resized;
            }
        }

        void rotate(const double ccw_rads, const cv::Mat* _src,
            cv::Mat* dst, PixelAndPose* map_origin_transformed) const
        {
            // rotation matrix
            const auto src = _src ? *_src : *image_;
            cv::Mat rot; cv::Size2f dsize;
            get_rotation_matrix_and_bounded_size(ccw_rads, src, rot, dsize);

            if(dst) {
                cv::warpAffine(src, *dst, rot, dsize,
                    cv::INTER_NEAREST, cv::BORDER_CONSTANT, unknown_pixel_value_);
            }

            if(map_origin_transformed) {
                // transform bottom-left corner co-ordinate and map origin pixel
                std::vector<cv::Point2f> vec {
                    cv::Point2f(0, src.rows),
                    cv::Point2f(-origin_[0] / res_, src.rows + (origin_[1] / res_))
                };
                cv::transform(vec, vec, rot);

                // transformed bottom-left corner co-ordinate
                map_origin_transformed->pt = vec[0];

                // new pose of transformed bottom-left corner co-ordinate
                map_origin_transformed->pos(1) = (vec[0].x - vec[1].x) * res_;
                map_origin_transformed->pos(1) = -(vec[0].y - vec[1].y) * res_;
                map_origin_transformed->pos(2) += ccw_rads;
            }
        }

        Masks get_masks(const cv::Mat* _image = nullptr) const
        {
            const auto *image = _image ? _image : image_;

            Masks masks;
            cv::Mat inv = yaml_node_["negate"].as<int>() ? *image : 255 - *image;
            cv::threshold(inv, masks.free, free_thresh_, 255, cv::THRESH_BINARY_INV);
            cv::threshold(inv, masks.occupied, occ_thresh_, 255, cv::THRESH_BINARY);
            masks.set_unknown_from_free_and_occupied();

            return masks;
        }

        void write(const Eigen::Vector3d& new_origin) const
        {
            auto node = yaml_node();

            node["origin"] = std::vector<double> { new_origin[0], new_origin[1], new_origin[2] };

            auto transform_node = node["transform"];
            if(transform_node) {
                transform_node = from_eigen(
                    from_xy_yaw(new_origin) * from_xy_yaw(origin_).inverse() * from_yaml_tf(transform_node)
                );
            }
               
            auto path = path_.parent_path() / (path_.stem().string() + "_aligned.yaml");
            write_yaml_node(node, path);
        }

        YAML::Node copy_metadata(bool include_map_if_available) const
        {
            YAML::Node node;

            node["resolution"] = res_;
            node["origin"] = YAML::Clone(yaml_node_["origin"]);
            node["occupied_thresh"] = YAML::Clone(yaml_node_["occupied_thresh"]);
            node["free_thresh"] = YAML::Clone(yaml_node_["free_thresh"]);
            node["negate"] = YAML::Clone(yaml_node_["negate"]);

            auto transform_node = node["transform"];
            if(transform_node && include_map_if_available)
            {
                node["map"] = YAML::Clone(yaml_node_["map"]);
                node["transform"] = YAML::Clone(yaml_node_["transform"]);
            }

            return node;
        }

        PixelAndPose origin() const
        {
            PixelAndPose pp;
            pp.pt.x = 0; pp.pt.y = image_->rows;
            pp.pos = Eigen::Vector3d(origin_);

            return pp;
        }

        YAML::Node yaml_node() const { return YAML::Clone(yaml_node_); }
        double res() const { return res_; }
        uint8_t unknown_pixel_value() const { return unknown_pixel_value_; }
        cv::Size size() const { return image_->size(); }

    private:
        std::filesystem::path path_;
        YAML::Node yaml_node_;
        cv::Mat *image_;
        Eigen::Vector3d origin_;
        double res_;
        uint8_t free_thresh_, occ_thresh_, unknown_pixel_value_;
};

struct MapWithOriginAndMasks
{
    std::shared_ptr<Map> map;
    Map::PixelAndPose origin;
    Map::Masks masks;

    MapWithOriginAndMasks(const std::string& yaml_fp, const double target_resolution=-1.0)
    {
        map = std::make_shared<Map>(yaml_fp, target_resolution);
        origin = map->origin(); masks = map->get_masks();
    }

    cv::Size size() const { return map->size(); }
};

#endif
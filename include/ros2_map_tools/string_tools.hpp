#ifndef ROS2_MAP_TOOLS_STRING_TOOLS_HPP
#define ROS2_MAP_TOOLS_STRING_TOOLS_HPP

#include <vector>
#include <string>

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

std::vector<double> split_string_to_doubles(const std::string& s, const char sep)
{
    std::vector<double> doubles;
    for(const auto& s : split_string(s, sep)) {
        doubles.push_back(std::atof(s.c_str()));
    }

    return doubles;
}

#endif
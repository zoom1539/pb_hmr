#pragma once

#include "opencv2/opencv.hpp"

class HMR
{ 
public:
    explicit HMR();
    ~HMR();

    bool serialize(const std::string &wts_path_, const std::string &engine_path_);

    // choose one from two
    // 1
    bool init(const std::string &engine_path_);
    bool run(const std::vector<cv::Mat> &imgs_, 
             std::vector<std::vector<cv::Vec3f> > &poses_,
             std::vector<std::vector<float> > &shapes_);
    
    // 2
    bool init_joints(const std::string &engine_path_, std::string &smpl_male_json_path_);
    bool run_joints(const std::vector<cv::Mat> &imgs_, 
             std::vector<std::vector<cv::Vec3f> > &poses_,
             std::vector<std::vector<float> > &shapes_,
             std::vector<std::vector<cv::Vec3f> > &vec_3djoints_,
             std::vector<std::vector<cv::Vec3f> > &vec_vertices_);
    

private:
    HMR(const HMR &);
    const HMR &operator=(const HMR &);

    class Impl;
    Impl *_impl;
};

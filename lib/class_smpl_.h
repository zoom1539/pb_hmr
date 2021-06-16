#pragma once

// std
#include <opencv2/opencv.hpp>
#include "smpl/SMPL.h"

class _SMPL
{
public:
    _SMPL();
    ~_SMPL();

public:
    bool init(std::string &smpl_male_json_path_);
    bool run(float theta[72], 
             float beta[10],
             std::vector<cv::Vec3f> &joints_,
             std::vector<cv::Vec3f> &vertices_);
    
    
private:
    smpl::SMPL *_smpl;

};

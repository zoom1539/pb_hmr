#pragma once

// std
#include <opencv2/opencv.hpp>
#include "NvInferRuntime.h"
#include "class_smpl_.h"

class _HMR
{
public:
    _HMR();
    ~_HMR();

public:
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
    nvinfer1::IRuntime* _runtime;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IExecutionContext* _context;
    cudaStream_t _stream;

    void* _buffers[2];

    //
    _SMPL _smpl;
};

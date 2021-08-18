#pragma once

// std
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferRuntime.h"

using namespace nvinfer1;

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
    

private:
    nvinfer1::IRuntime* _runtime;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IExecutionContext* _context;
    cudaStream_t _stream;

    void* _buffers[2];


};

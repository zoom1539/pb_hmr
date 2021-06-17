#pragma once

// std
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "class_smpl_.h"

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
    
    // 2
    bool init_joints(const std::string &engine_path_, std::string &smpl_male_json_path_);
    bool run_joints(const std::vector<cv::Mat> &imgs_, 
             std::vector<std::vector<cv::Vec3f> > &poses_,
             std::vector<std::vector<float> > &shapes_,
             std::vector<std::vector<cv::Vec3f> > &vec_3djoints_,
             std::vector<std::vector<cv::Vec3f> > &vec_vertices_);
// private:
//     void rot6d_to_mat(const cv::Mat &rot6d_, cv::Mat &rotmat_);
//     std::map<std::string, Weights> loadWeights(const std::string file);
//     nvinfer1::IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
//     nvinfer1::IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname);
//     nvinfer1::ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config,DataType dt, const std::string &wts_path_);
//     void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string &wts_path_);
//     void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
//     cv::Mat get_transform(const cv::Point2i &center_, float scale_, int res_);
//     cv::Point2i transform(const cv::Point2i &pt_, const cv::Point2i &center_, float scale_, int res_);
//     void preprocess_img(const cv::Mat &img_, cv::Mat &img_preprocess_);

private:
    nvinfer1::IRuntime* _runtime;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IExecutionContext* _context;
    cudaStream_t _stream;

    void* _buffers[2];

    //
    _SMPL _smpl;
};

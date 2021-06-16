#include "class_smpl_.h"
#include <torch/torch.h>
#include "definition/def.h"
#include "toolbox/TorchEx.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xjson.hpp>

_SMPL::_SMPL() {}
_SMPL::~_SMPL() 
{
   if (_smpl)
   {
       delete _smpl;
       _smpl = nullptr;
   }
}

bool _SMPL::init(std::string &smpl_male_json_path_)
{
    torch::Device cuda(torch::kCUDA);
	cuda.set_index(0);
    
    _smpl = new smpl::SMPL();
    _smpl->setDevice(cuda);
    _smpl->setModelPath(smpl_male_json_path_);
    _smpl->init();

    return true;
}

bool _SMPL::run(float theta_[72], 
                float beta_[10],
                std::vector<cv::Vec3f> &joints_,
                std::vector<cv::Vec3f> &vertices_)
{
    torch::Tensor beta = torch::from_blob(beta_, {1, SHAPE_BASIS_DIM });
	torch::Tensor theta = torch::from_blob(theta_, {1, JOINT_NUM, 3});
    
    //
    std::vector<float> joints;
    _smpl->launch(beta, theta, joints);
    for (int i = 0; i < JOINT_NUM; i++)
    {
        cv::Vec3f joint;
        joint[0] = joints[i * 3];
        joint[1] = joints[i * 3 + 1];
        joint[2] = joints[i * 3 + 2];
        joints_.push_back(joint);
    }
    

    //
    torch::Tensor vertices = _smpl->getVertex().clone();
    torch::Tensor slice_ = smpl::TorchEx::indexing(vertices,
                                             torch::IntList({0}));// (6890, 3)
    xt::xarray<float> slice = xt::adapt(
        (float *)slice_.to(torch::kCPU).data_ptr(),
                           xt::xarray<float>::shape_type({(const size_t)VERTEX_NUM, 3}));

    for (int i = 0; i < VERTEX_NUM; i++) 
    {
        cv::Vec3f vertice;
        vertice[0] = slice(i, 0);
        vertice[1] = slice(i, 1);
        vertice[2] = slice(i, 2);
        vertices_.push_back(vertice);
    }
}

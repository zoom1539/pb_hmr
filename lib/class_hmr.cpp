#include "class_hmr.h"
#include "class_hmr_.h"

class HMR::Impl
{
public:
    _HMR _hmr;
};

HMR::HMR() : _impl(new HMR::Impl())
{
}

HMR::~HMR()
{
    delete _impl;
    _impl = NULL;
}

bool HMR::serialize(const std::string &wts_path_, const std::string &engine_path_)
{
    return _impl->_hmr.serialize(wts_path_, engine_path_);
}

bool HMR::init(const std::string &engine_path_)
{
    return _impl->_hmr.init(engine_path_);
}

bool HMR::run(const std::vector<cv::Mat> &imgs_, 
             std::vector<std::vector<cv::Vec3f> > &poses_,
             std::vector<std::vector<float> > &shapes_)
{
    return _impl->_hmr.run(imgs_, poses_, shapes_);
}

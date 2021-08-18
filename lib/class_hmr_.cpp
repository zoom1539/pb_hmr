#include "class_hmr_.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include "logging.h"

using namespace nvinfer1;

static Logger gLogger;

#define USE_FP16
#define DEVICE (0)
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int BATCH_SIZE = 1;
static const int POSE_SIZE = 144;
static const int SHAPE_SIZE = 10;
static const int CAM_SIZE = 3;
static const int OUTPUT_SIZE = POSE_SIZE + SHAPE_SIZE;
static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";

static void rot6d_to_mat(const cv::Mat &rot6d_, cv::Mat &rotmat_)
{
    // TEST
    // cv::Mat rot6d = (cv::Mat_<float>(1,6) << 0.4250,  0.0280, 0.0810, -1.0976, -0.7952, -0.0642);
    // cv::Mat rotmat;
    // rot6d_to_mat(rot6d, rotmat);
    // std::cout << rotmat;
    // [ 0.4695,  0.0377, -0.8821],
    // [ 0.0895, -0.9960,  0.0050],
    // [-0.8784, -0.0813, -0.4709]
    
    float a00 = rot6d_.at<float>(0,0);
    float a01 = rot6d_.at<float>(0,1);
    float a10 = rot6d_.at<float>(0,2);
    float a11 = rot6d_.at<float>(0,3);
    float a20 = rot6d_.at<float>(0,4);
    float a21 = rot6d_.at<float>(0,5);
    // col 0
    float norm = std::sqrt(a00 * a00 + a10 * a10 + a20 * a20);
    float b00 = a00 / (norm + FLT_EPSILON);
    float b10 = a10 / (norm + FLT_EPSILON);
    float b20 = a20 / (norm + FLT_EPSILON);

    // col 1
    float c = b00 * a01 + b10 * a11 + b20 * a21;
    float d0 = a01 - c * b00;
    float d1 = a11 - c * b10;
    float d2 = a21 - c * b20;

    norm = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    float b01 = d0 / (norm + FLT_EPSILON);
    float b11 = d1 / (norm + FLT_EPSILON);
    float b21 = d2 / (norm + FLT_EPSILON);

    // col 2
    cv::Vec3f b0(b00, b10, b20);
    cv::Vec3f b1(b01, b11, b21);
    cv::Vec3f b2 = b0.cross(b1);

    // output 
    rotmat_ = (cv::Mat_<float>(3,3) << b00, b01, b2[0],
                                       b10, b11, b2[1],
                                       b20, b21, b2[2]);

}

static std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count = 0;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

static IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) 
{
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

static IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

static ICudaEngine* createEngine(unsigned int maxBatchSize, 
                          IBuilder* builder, 
                          IBuilderConfig* config, 
                          DataType dt, 
                          const std::string &wts_path_)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_path_);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.3.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.5.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.2.");

    IPoolingLayer* pool2 = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});

    //std::cout << "resnet50 create" << std::endl;

    //
    IConstantLayer *init_pose = network->addConstant(Dims3{144, 1, 1}, weightMap["init_pose"]);
    IConstantLayer *init_shape = network->addConstant(Dims3{10, 1, 1}, weightMap["init_shape"]);
    IConstantLayer *init_cam = network->addConstant(Dims3{3, 1, 1}, weightMap["init_cam"]);

    std::cout << "init param create" << std::endl;


    // 1
    ITensor* IEFinputTensors1[] = {pool2->getOutput(0), init_pose->getOutput(0), init_shape->getOutput(0), init_cam->getOutput(0)};
    IConcatenationLayer* IEFcat1 = network->addConcatenation(IEFinputTensors1, 4);
    assert(IEFcat1);

    IFullyConnectedLayer* fc1_1 = network->addFullyConnected(*IEFcat1->getOutput(0), 1024, weightMap["fc1.weight"], weightMap["fc1.bias"]);
    assert(fc1_1);
    IFullyConnectedLayer* fc1_2 = network->addFullyConnected(*fc1_1->getOutput(0), 1024, weightMap["fc2.weight"], weightMap["fc2.bias"]);
    assert(fc1_2);
    
    IFullyConnectedLayer* decpose1 = network->addFullyConnected(*fc1_2->getOutput(0), 144, weightMap["decpose.weight"], weightMap["decpose.bias"]);
    assert(decpose1);
    IElementWiseLayer* pred_pose1 = network->addElementWise(*decpose1->getOutput(0), *init_pose->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_pose1);
    
    IFullyConnectedLayer* decshape1 = network->addFullyConnected(*fc1_2->getOutput(0), 10, weightMap["decshape.weight"], weightMap["decshape.bias"]);
    assert(decshape1);
    IElementWiseLayer* pred_shape1 = network->addElementWise(*decshape1->getOutput(0), *init_shape->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_shape1);
    
    IFullyConnectedLayer* deccam1 = network->addFullyConnected(*fc1_2->getOutput(0), 3, weightMap["deccam.weight"], weightMap["deccam.bias"]);
    assert(deccam1);
    IElementWiseLayer* pred_cam1 = network->addElementWise(*deccam1->getOutput(0), *init_cam->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_cam1);

    //std::cout << "IEF1 create" << std::endl;

    // 2
    ITensor* IEFinputTensors2[] = {pool2->getOutput(0), pred_pose1->getOutput(0), pred_shape1->getOutput(0), pred_cam1->getOutput(0)};
    IConcatenationLayer* IEFcat2 = network->addConcatenation(IEFinputTensors2, 4);
    assert(IEFcat2);

    IFullyConnectedLayer* fc2_1 = network->addFullyConnected(*IEFcat2->getOutput(0), 1024, weightMap["fc1.weight"], weightMap["fc1.bias"]);
    assert(fc2_1);
    IFullyConnectedLayer* fc2_2 = network->addFullyConnected(*fc2_1->getOutput(0), 1024, weightMap["fc2.weight"], weightMap["fc2.bias"]);
    assert(fc2_2);
    
    IFullyConnectedLayer* decpose2 = network->addFullyConnected(*fc2_2->getOutput(0), 144, weightMap["decpose.weight"], weightMap["decpose.bias"]);
    assert(decpose2);
    IElementWiseLayer* pred_pose2 = network->addElementWise(*decpose2->getOutput(0), *pred_pose1->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_pose2);
    
    IFullyConnectedLayer* decshape2 = network->addFullyConnected(*fc2_2->getOutput(0), 10, weightMap["decshape.weight"], weightMap["decshape.bias"]);
    assert(decshape2);
    IElementWiseLayer* pred_shape2 = network->addElementWise(*decshape2->getOutput(0), *pred_shape1->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_shape2);
    
    IFullyConnectedLayer* deccam2 = network->addFullyConnected(*fc2_2->getOutput(0), 3, weightMap["deccam.weight"], weightMap["deccam.bias"]);
    assert(deccam2);
    IElementWiseLayer* pred_cam2 = network->addElementWise(*deccam2->getOutput(0), *pred_cam1->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_cam2);

    //std::cout << "IEF2 create" << std::endl;


    // 3
    ITensor* IEFinputTensors3[] = {pool2->getOutput(0), pred_pose2->getOutput(0), pred_shape2->getOutput(0), pred_cam2->getOutput(0)};
    IConcatenationLayer* IEFcat3 = network->addConcatenation(IEFinputTensors3, 4);
    assert(IEFcat3);

    IFullyConnectedLayer* fc3_1 = network->addFullyConnected(*IEFcat3->getOutput(0), 1024, weightMap["fc1.weight"], weightMap["fc1.bias"]);
    assert(fc3_1);
    IFullyConnectedLayer* fc3_2 = network->addFullyConnected(*fc3_1->getOutput(0), 1024, weightMap["fc2.weight"], weightMap["fc2.bias"]);
    assert(fc3_2);
    
    IFullyConnectedLayer* decpose3 = network->addFullyConnected(*fc3_2->getOutput(0), 144, weightMap["decpose.weight"], weightMap["decpose.bias"]);
    assert(decpose3);
    IElementWiseLayer* pred_pose3 = network->addElementWise(*decpose3->getOutput(0), *pred_pose2->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_pose3);
    
    IFullyConnectedLayer* decshape3 = network->addFullyConnected(*fc3_2->getOutput(0), 10, weightMap["decshape.weight"], weightMap["decshape.bias"]);
    assert(decshape3);
    IElementWiseLayer* pred_shape3 = network->addElementWise(*decshape3->getOutput(0), *pred_shape2->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_shape3);
    
    IFullyConnectedLayer* deccam3 = network->addFullyConnected(*fc3_2->getOutput(0), 3, weightMap["deccam.weight"], weightMap["deccam.bias"]);
    assert(deccam3);
    IElementWiseLayer* pred_cam3 = network->addElementWise(*deccam3->getOutput(0), *pred_cam2->getOutput(0), ElementWiseOperation::kSUM);
    assert(pred_cam3);

    //std::cout << "IEF3 create" << std::endl;

    ITensor* output_tensors[] = {pred_pose3->getOutput(0), pred_shape3->getOutput(0)};
    IConcatenationLayer* output_cat = network->addConcatenation(output_tensors, 2);

    output_cat->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*output_cat->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

static void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, const std::string &wts_path_)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wts_path_);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

static void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

static cv::Mat get_transform(const cv::Point2i &center_, float scale_, int res_)
{
    float h = 200 * scale_;
    cv::Mat t = cv::Mat::eye(3,3,CV_32F);
    t.at<float>(0,0) = res_ / h;
    t.at<float>(1,1) = res_ / h;
    t.at<float>(0,2) = res_ * (- center_.x / h + 0.5);
    t.at<float>(1,2) = res_ * (- center_.y / h + 0.5);
    return t;
}

static cv::Point2i transform(const cv::Point2i &pt_, const cv::Point2i &center_, float scale_, int res_)
{
    cv::Mat t  = get_transform(center_, scale_, res_);
    cv::Mat t_i = t.inv();
    cv::Mat new_pt = (cv::Mat_<float>(3,1) << pt_.x - 1, pt_.y - 1, 1.0);
    new_pt = t_i * new_pt;

    return cv::Point2i(new_pt.at<float>(0,0) + 1, new_pt.at<float>(1,0) + 1);
}

static void preprocess_img(const cv::Mat &img_, cv::Mat &img_preprocess_)
{
    int height = img_.rows;
    int width = img_.cols;
    cv::Point2i center(width * 0.5, height * 0.5);
    float scale = std::max(height, width) / 200.0;
    int res = 224;

    cv::Point2i ul = transform(cv::Point2i(1,1), center, scale, res) - cv::Point2i(1,1);
    cv::Point2i br = transform(cv::Point2i(res + 1,res + 1), center, scale, res) - cv::Point2i(1,1);
    
    cv::Mat new_img = cv::Mat::zeros(cv::Size( br.x - ul.x, br.y - ul.y), CV_8UC3);

    cv::Vec2i new_x(std::max(0, -ul.x), std::min(br.x, width)-ul.x);
    cv::Vec2i new_y(std::max(0, -ul.y), std::min(br.y, height)-ul.y);
    cv::Vec2i old_x(std::max(0, ul.x), std::min(width, br.x));
    cv::Vec2i old_y(std::max(0, ul.y), std::min(height, br.y));

    cv::Mat roi = new_img(cv::Rect(new_x[0], new_y[0], new_x[1]- new_x[0], new_y[1]-new_y[0]));

    img_(cv::Rect(old_x[0], old_y[0], old_x[1]- old_x[0], old_y[1]-old_y[0])).copyTo(roi);

    cv::Mat img_preprocess = cv::Mat(INPUT_H, INPUT_W, CV_8UC3);
    cv::resize(new_img, img_preprocess, img_preprocess.size());

    img_preprocess_ = img_preprocess.clone();
}

_HMR::_HMR() {}
_HMR::~_HMR() 
{
    if (_buffers[0])
    {
        CUDA_CHECK(cudaFree(_buffers[0]));
    }
    if (_buffers[1])
    {
        CUDA_CHECK(cudaFree(_buffers[1]));
    }
    
    // Destroy the engine
    if (_context)
    {
        _context->destroy();
    }
    
    if (_engine)
    {
        _engine->destroy();
    }
    
    if (_runtime)
    {
        _runtime->destroy();
    }
}

bool _HMR::serialize(const std::string &wts_path_, const std::string &engine_path_)
{
    IHostMemory* modelStream{nullptr};
    APIToModel(BATCH_SIZE, &modelStream, wts_path_);
    assert(modelStream != nullptr);

    std::ofstream p(engine_path_, std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return false;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return true;
}
 

bool _HMR::init(const std::string &engine_path_)
{
    cudaSetDevice(DEVICE);

    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_path_ << " error!" << std::endl;
        return false;
    }

    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    _runtime = createInferRuntime(gLogger);
    assert(_runtime != nullptr);
    _engine = _runtime->deserializeCudaEngine(trtModelStream, size);
    assert(_engine != nullptr);
    _context = _engine->createExecutionContext();
    assert(_context != nullptr);
    delete[] trtModelStream;
    assert(_engine->getNbBindings() == 2);
    
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = _engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = _engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&_buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&_stream));

    return true;
}

bool _HMR::run(const std::vector<cv::Mat> &imgs_, 
               std::vector<std::vector<cv::Vec3f> > &poses_,
               std::vector<std::vector<float> > &shapes_)
{
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    static float output_param[BATCH_SIZE * OUTPUT_SIZE];


    int fcount = 0;
    for (int f = 0; f < (int)imgs_.size(); f++) 
    {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)imgs_.size()) 
        {
            continue;
        }

        // auto start1 = std::chrono::system_clock::now();

        for (int b = 0; b < fcount; b++) 
        {
            cv::Mat img = imgs_[f - fcount + 1 + b];
            
            // 
            cv::Mat img_preprocess;
            preprocess_img(img, img_preprocess);
                
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) 
            {
                uchar* uc_pixel = img_preprocess.data + row * img_preprocess.step;
                for (int col = 0; col < INPUT_W; ++col) 
                {
                    data[b * 3 * INPUT_H * INPUT_W + i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // auto end1 = std::chrono::system_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;
        

        // Run inference
        // auto start = std::chrono::system_clock::now();
        
        doInference(*_context, _stream, _buffers, data, output_param, BATCH_SIZE);
        
        // auto end = std::chrono::system_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ---ms" << std::endl;
        
        for (int b = 0; b < fcount; b++) 
        {
            //
            std::vector<cv::Vec3f> pose;
            for (int i = 0; i < OUTPUT_SIZE - SHAPE_SIZE; i+=6)
            {
                cv::Mat rot6d = (cv::Mat_<float>(1,6) << output_param[b * OUTPUT_SIZE + i],  
                                                         output_param[b * OUTPUT_SIZE + i + 1],
                                                         output_param[b * OUTPUT_SIZE + i + 2],
                                                         output_param[b * OUTPUT_SIZE + i + 3],
                                                         output_param[b * OUTPUT_SIZE + i + 4],
                                                         output_param[b * OUTPUT_SIZE + i + 5]
                                                         );
                cv::Mat rotmat;
                rot6d_to_mat(rot6d, rotmat);
                cv::Vec3f axis_angle;
                cv::Rodrigues(rotmat, axis_angle);
                pose.push_back(axis_angle);
            }
            
            poses_.push_back(pose);

            //
            std::vector<float> shape;
            for (int i = 0; i < SHAPE_SIZE; i++)
            {
                shape.push_back(output_param[b * OUTPUT_SIZE + POSE_SIZE + i]);
            }
            shapes_.push_back(shape);
        }

        //
        fcount = 0;
    }

    return true;
}

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <chrono>
#include "class_hmr.h"

int main()
{
    HMR hmr;

    //
    std::string wts_path = "../lib/extra/hmr.wts";
    std::string engine_path = "../lib/extra/hmr_fp16_b1.engine";
#if 1
    bool is_serialized = hmr.serialize(wts_path, engine_path);
    if(!is_serialized)
    {
        std::cout << "serialize fail\n";
        return 0;
    }
#else

    bool is_init = hmr.init(engine_path);
    if(!is_init)
    {
        std::cout << "serialize fail\n";
        return 0;
    }

    std::vector<cv::Mat> imgs;
    {
        // cv::Mat img = cv::Mat(cv::Size(224, 224), CV_8UC3, cv::Scalar(255,255,255));
        cv::Mat img = cv::imread("../data/my_test_1.jpg");
        imgs.push_back(img);
    }
    
    auto start = std::chrono::system_clock::now();
    std::vector<std::vector<cv::Vec3f> > poses;
    std::vector<std::vector<float> > shapes;
    bool is_run = hmr.run(imgs, poses, shapes);

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " total ms" << std::endl;

    for (int i = 0; i < poses.size(); i++)
    {
        for (int j = 0; j < poses[i].size(); j++)
        {
            std::cout << poses[i][j] << std::endl;
        }
    }

    for (int i = 0; i < shapes.size(); i++)
    {
        for (int j = 0; j < shapes[i].size(); j++)
        {
            std::cout << shapes[i][j] << std::endl;
        }
    }

#endif

    std::cin.get();
    return 0;
}


/*******************************************************
 文件名：Yolov11Pose.h
 作者：
 描述：Yolov11姿态估计推理模块
 版本：v1.0
 日期：2025-05-09
 *******************************************************/

#pragma once
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "IBaseModule.h"
#include "CMultiModalSrcData.h"
#include "PoseEstimation_conf.pb.h"
#include "glog/logging.h"

namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}
extern "C" {
    typedef struct CUstream_st* cudaStream_t;
}

class Yolov11Pose : public IBaseModule {
public:
    Yolov11Pose(const std::string& exe_path);
    ~Yolov11Pose() override;

    std::string getModuleName() const override { return "Yolov11Pose"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(void* p_pAlgParam) override;
    void* execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

    void cleanup();
    std::vector<float> inference(const cv::Mat& img);
    void preprocess(const cv::Mat& img, std::vector<float>& input);
    void rescale_coords(std::vector<float>& coords, bool is_keypoint);
    std::vector<std::vector<float>> process_keypoints(const std::vector<float>& output, const std::vector<std::vector<float>>& boxes);
    std::vector<std::vector<float>> process_output(const std::vector<float>& output);
    std::vector<int> nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores);

private:
    PoseConfig m_poseConfig;
    cv::Mat m_inputImage;
    // 假设最终输出结构体为CAlgResult
    CAlgResult m_outputResult;

    // TensorRT相关成员
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> input_buffers_;
    std::vector<void*> output_buffers_;
    cudaStream_t stream_ = nullptr;

    // 后处理参数
    int new_unpad_h_;
    int new_unpad_w_;
    int dw_;
    int dh_;
    float ratio_;
    int stride_;
    int batch_size_;
    float conf_thres_;
    float iou_thres_;
    int num_classes_;
};
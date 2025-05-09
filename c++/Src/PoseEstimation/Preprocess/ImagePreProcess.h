/*******************************************************
 文件名：ImagePreProcess.h
 作者：
 描述：图像预处理模块
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef IMAGE_PRE_PROCESS_H
#define IMAGE_PRE_PROCESS_H


#include <opencv2/opencv.hpp>
#include <iostream>
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "CMultiModalSrcData.h"
#include "PoseEstimation_conf.pb.h"
#include "glog/logging.h"

class ImagePreProcess : public IBaseModule {
public:
    ImagePreProcess(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~ImagePreProcess() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ImagePreProcess"; }
    ModuleType getModuleType() const override { return ModuleType::PRE_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void* execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
   PoseConfig m_poseConfig;             // 姿态估计任务配置参数
   CMultiModalSrcData m_inputImage;     // 预处理输入数据
   cv::Mat m_outputImage;               // 预处理输出数据
};

#endif // IMAGE_PRE_PROCESS_H 
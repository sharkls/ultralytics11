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
#include <vector>
#include <algorithm>
#include "log.h"
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "CAlgResult.h"
#include "PoseEstimation_conf.pb.h"
#include "../CPoseEstimationAlg.h"


class ImagePreProcess : public IBaseModule {
public:
    ImagePreProcess(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~ImagePreProcess() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ImagePreProcess"; }
    ModuleType getModuleType() const override { return ModuleType::PRE_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
    // 处理单个子图的预处理
    std::vector<float> processSingleImage(const cv::Mat& srcImage, int& outWidth, int& outHeight);
    
    // 处理单个子图的预处理（带填充，用于批量处理）
    std::vector<float> processSingleImageWithPadding(const cv::Mat& srcImage, int targetWidth, int targetHeight, 
                                                     float& ratio, int& padTop, int& padLeft);

   posetimation::YOLOModelConfig m_poseConfig;            // 姿态估计任务配置参数
   CAlgResult m_inputData;                                // 输入数据（CAlgResult格式）
   MultiImagePreprocessResult m_outputResult;             // 多图像预处理结果

   // 图像相关参数
   int max_model_size_;  // 模型输入最大尺寸
   int stride_;          // 模型最大步长

   // 运行状态
   bool status_ = false;
};

#endif // IMAGE_PRE_PROCESS_H 
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
#include "log.h"
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "ObjectDetection_conf.pb.h"


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

   objectdetection::YOLOModelConfig m_poseConfig;            // 目标检测任务配置参数
   CMultiModalSrcData m_inputImage;         // 预处理输入数据
   std::vector<float> m_outputImage;        // 模型输入数据缓存区

   // 图像相关参数
   int src_w_;  // 原始图像宽度
   int src_h_;  // 原始图像长度
   int max_model_size_;  // 模型输入最大尺寸
   int new_unpad_w_;     // 等比缩放并填充后的宽度
   int new_unpad_h_;     // 等比缩放并填充后的高度
   int dw_;              // 左右填充
   int dh_;              // 上下填充
   int stride_;          // 模型最大步长

   // 运行状态
   bool status_ = false;
};

#endif // IMAGE_PRE_PROCESS_H 
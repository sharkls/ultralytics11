/*******************************************************
 文件名：ImagePreProcess.h
 作者：sharkls
 描述：图像预处理模块
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef IMAGE_PRE_PROCESS_H
#define IMAGE_PRE_PROCESS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "log.h"
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "MultiModalFusion_conf.pb.h"


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

    // 返回预处理变换矩阵（3x3）
    cv::Mat preprocess(cv::Mat img, int new_pad_w, int new_pad_h, int dw, int dh, float r, std::vector<float>& output);

private:
   multimodalfusion::MultiModalFusionModelConfig m_config;  // 任务配置参数
   CMultiModalSrcData m_inputImage;                         // 预处理输入数据
   std::vector<std::vector<float>> m_outputImage;           // 模型输入数据缓存区

   // 图像相关参数（分别为可见光和红外）
   int src_w_rgb_, src_h_rgb_, new_unpad_w_rgb_, new_unpad_h_rgb_, dw_rgb_, dh_rgb_;
   int src_w_ir_,  src_h_ir_,  new_unpad_w_ir_,  new_unpad_h_ir_,  dw_ir_,  dh_ir_;
   float r_rgb_, r_ir_;
   int stride_;
   int max_model_size_;

   // 运行状态
   bool status_ = false;
};

#endif // IMAGE_PRE_PROCESS_H 
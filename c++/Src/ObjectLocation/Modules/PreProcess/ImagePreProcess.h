/*******************************************************
 文件名：ImagePreProcess.h
 作者：
 描述：图像预处理模块
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef IMAGE_PRE_PROCESS_H
#define IMAGE_PRE_PROCESS_H

#include "../../../Common/IBaseModule.h"
#include <opencv2/opencv.hpp>

class ImagePreProcess : public IBaseModule {
public:
    ImagePreProcess();
    ~ImagePreProcess() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ImagePreProcess"; }
    ModuleType getModuleType() const override { return ModuleType::PRE_PROCESS; }
    bool init(CSelfAlgParam* p_pAlgParam) override;
    void* execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
    // 图像预处理参数
    struct PreProcessParams {
        int targetWidth = 640;
        int targetHeight = 640;
        float mean[3] = {0.485f, 0.456f, 0.406f};
        float std[3] = {0.229f, 0.224f, 0.225f};
    };

    PreProcessParams m_params;
    cv::Mat m_inputImage;
    cv::Mat m_outputImage;
};

#endif // IMAGE_PRE_PROCESS_H 
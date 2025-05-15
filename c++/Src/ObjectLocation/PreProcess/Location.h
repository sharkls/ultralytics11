/*******************************************************
 文件名：Location.h
 作者：
 描述：目标定位预处理模块
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef LOCATION_H
#define LOCATION_H


#include <opencv2/opencv.hpp>
#include <iostream>
#include "log.h"

#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"

#include "CAlgResult.h"
#include "ObjectLocation_conf.pb.h"


class Location : public IBaseModule {
public:
    Location(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~Location() override;

    // 实现基类接口
    std::string getModuleName() const override { return "Location"; }
    ModuleType getModuleType() const override { return ModuleType::PRE_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:

   objectlocation::ObjectLocationConfig m_config;            // 目标定位任务配置参数
   CAlgResult m_inputdata;                                   // 预处理输入数据
   CAlgResult m_outputdata;                                  // 预处理输出数据

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

#endif // LOCATION_H 
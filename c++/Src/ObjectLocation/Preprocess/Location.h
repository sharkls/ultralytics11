/*******************************************************
 文件名：Location.h
 作者：sharkls
 描述：目标定位预处理模块
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#ifndef LOCATION_H
#define LOCATION_H


#include <opencv2/opencv.hpp>
#include <iostream>
#include "log.h"
#include <vector>
#include <algorithm>
#include <map>
#include <cmath>

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
    // 辅助函数
    float calc_iou(const CObjectResult& a, const CObjectResult& b) const;
    float get_depth(const std::vector<float>& depth_map, int width, int height, float x, float y) const;
    float get_bucket_depth(const std::vector<float>& depths, float bucket_size = 5.0f) const;

    objectlocation::TaskConfig m_config;            // 目标定位任务配置参数
    CAlgResult m_inputdata;                                   // 预处理输入数据
    CAlgResult m_outputdata;                                  // 预处理输出数据

    // 图像相关参数
    float iou_thres_;       // iou阈值
    int num_keys_;          // 关键点数量
    float bucket_size_;     // 桶大小
    float max_distance_;    // 最大距离

    // 运行状态
    bool status_ = false;
};

#endif // LOCATION_H 
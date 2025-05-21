/*******************************************************
 文件名：ExportPoseEstimationAlgLib.h
 作者：
 描述：姿态估计算法库的算法接口类导出函数头文件
 版本：v1.0
 日期：2025-04-27
 *******************************************************/
#pragma once
#include <string>

#include "CSelfAlgParam.h"

struct IPoseEstimationAlg
{
    IPoseEstimationAlg(){};
    virtual ~IPoseEstimationAlg(){};

    //初始化算法接口对象，内部主要处理只需初始化一次的操作，比如模型加载之类的，成功返回true，失败返回false
    virtual bool initAlgorithm(CSelfAlgParam* p_pAlgParam,  const AlgCallback& alg_cb, void* hd)   = 0;

    //执行算法函数，传入原始数据体，算法执行成功返回处理后的数据或者检测结果（由算法类型而定），失败返回nullptr
    virtual void runAlgorithm(void* p_pSrcData)  = 0;
};

extern "C" __attribute__ ((visibility("default"))) IPoseEstimationAlg*   CreatePoseEstimationAlgObj(const std::string& p_strExePath);
/*******************************************************
 文件：CSelfAlgParam.h
 作者：
 描述：算法内部使用的参数结构体
 版本：v2.0
 日期：2024-1-3
 *******************************************************/
#pragma once
#include <functional>
#include "CAlgResult.h"

using AlgCallback = std::function<void(const CAlgResult&, void*)>;

struct CSelfAlgParam
{
	CSelfAlgParam() {}
	virtual ~CSelfAlgParam(){}

	std::string					m_strRootPath;			//算法配置文件根目录路径
	std::string 				m_strEnginepath;
	float 						m_fConfidenceThreshold;
	float 						m_fNmsThreshold;
	int 						m_nNumClasses;
	int 						m_nSrcInputHeight;
	int 						m_nSrcInputWidth;
	int 						m_nResizeOutputHeight;
	int 						m_nResizeOutputWidth;
	int 						m_nBatchSize;
	int 						m_nInputChannels;
	int32_t						m_nNumAnchors;


};

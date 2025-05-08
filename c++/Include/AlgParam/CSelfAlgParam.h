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
	std::string 				m_strEnginepath;		//模型路径
	float 						m_fConfidenceThreshold;	//置信度阈值
	float 						m_fNmsThreshold;		//NMS阈值
	int 						m_nNumClasses;			//类别数量
	int 						m_nSrcInputHeight;		//输入高度
	int 						m_nSrcInputWidth;		//输入宽度
	int 						m_nResizeOutputHeight;	//预处理后的输出高度
	int 						m_nResizeOutputWidth;	//预处理后的输出宽度
	int 						m_nBatchSize;			//批量大小
	int 						m_nInputChannels;		//输入通道数
	int32_t						m_nNumAnchors;			//锚框数量
	int 						m_nMaxDetections;		//最大检测数量
};

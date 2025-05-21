/*******************************************************
 文件：CSelfAlgParam.h
 作者：sharkls
 描述：算法内部使用的参数结构体
 版本：v2.0
 日期：2025-05-15
 *******************************************************/
#pragma once
#include <functional>
#include "activities/idl/CAlgResult/CAlgResult.h"

// 算法接收的回调函数定义
using AlgCallback = std::function<void(const CAlgResult&, void*)>;

struct CSelfAlgParam
{
	CSelfAlgParam() {}
	virtual ~CSelfAlgParam(){}

	std::string					m_strRootPath;			//算法配置文件根目录路径
};

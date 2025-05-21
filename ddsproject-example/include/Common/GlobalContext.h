/*******************************************************
 文件：GlobalContext.h
 作者：sharkls
 描述：全局上下文
 版本：v2.0
 日期：2025-05-15
 *******************************************************/
#ifndef _GLOBAL_CONTEXT__H_
#define _GLOBAL_CONTEXT__H_

#include <stdint.h>
#include <string>

// ================== 时间戳类型 ==================
enum TIMESTAMP_TYPE : uint8_t
{
    // 红外图像相关
    TIMESTAMP_IR_EXPOSURE,      // 红外图像曝光时间戳
    TIMESTAMP_IR_ARRIVE,        // 红外图像落地（接收）时间戳
    TIMESTAMP_IR,               // 通用红外时间戳

    // 可见光图像相关
    TIMESTAMP_RGB_EXPOSURE,     // 可见光图像曝光时间戳
    TIMESTAMP_RGB_ARRIVE,       // 可见光图像落地（接收）时间戳
    TIMESTAMP_RGB,              // 通用可见光时间戳

    // 毫米波相关
    TIMESTAMP_RADAR,            // 毫米波时间戳

    // 时间匹配相关
    TIMESTAMP_TIME_MATCH,       // 用于时间匹配的时间戳

    // 多模态感知算法相关
    TIMESTAMP_MMALG_BEGIN,      // 进入多模态感知算法前时间戳
    TIMESTAMP_MMALG_END,        // 多模态感知算法后时间戳

    // 姿态估计算法相关
    TIMESTAMP_POSEALG_BEGIN,    // 进入姿态估计算法前时间戳
    TIMESTAMP_POSEALG_END,      // 姿态估计算法后时间戳

    // 目标定位算法相关
    TIMESTAMP_LOCALG_BEGIN,     // 进入目标定位算法前时间戳
    TIMESTAMP_LOCALG_END,       // 目标定位算法后时间戳

    TIMESTAMP_TYPE_MAX          // 时间戳类型总个数
};

// ================== 帧率类型 ==================
enum EFPSType : uint8_t
{
    FPS_IR_EXPOSURE,         // 红外图像曝光帧率
    FPS_IR_ARRIVE,           // 红外图像落地帧率
    FPS_RGB_EXPOSURE,        // 可见光图像曝光帧率
    FPS_RGB_ARRIVE,          // 可见光图像落地帧率
    FPS_RADAR,               // 毫米波数据帧率
    FPS_TIME_MATCH,          // 时间匹配帧率
    FPS_MMALG_INPUT,         // 进入多模态感知算法帧率
    FPS_MMALG_OUTPUT,        // 多模态感知算法输出帧率
    FPS_POSEALG_INPUT,       // 进入姿态估计算法帧率
    FPS_POSEALG_OUTPUT,      // 姿态估计算法输出帧率
    FPS_LOCALG_INPUT,        // 进入目标定位算法帧率
    FPS_LOCALG_OUTPUT,       // 目标定位算法输出帧率
    FPS_TYPE_MAX
};

// ================== 延时类型 ==================
enum EDelayType : uint8_t
{
    DELAY_TYPE_IR_EXPOSURE_TO_ARRIVE,      // 红外曝光到落地延时
    DELAY_TYPE_RGB_EXPOSURE_TO_ARRIVE,     // 可见光曝光到落地延时
    DELAY_TYPE_RADAR,                      // 毫米波数据处理延时
    DELAY_TYPE_TIME_MATCH,                 // 时间匹配耗时
    DELAY_TYPE_MMALG,                      // 多模态感知算法总耗时
    DELAY_TYPE_POSEALG,                    // 姿态估计算法总耗时
    DELAY_TYPE_LOCALG,                     // 目标定位算法总耗时
    DELAY_TYPE_PIPELINE_ALL,               // 全流程总延时
    DELAY_TYPE_MAX
};

// ================== 数据来源类型 ==================
enum DATA_SOURCE_TYPE : uint8_t
{
    DATA_SOURCE_ONLINE,    // 在线
    DATA_SOURCE_OFFLINE,   // 离线
    DATA_SOURCE_SIM,       // 仿真/回放
    DATA_SOURCE_TYPE_MAX
};

// ================== 数据类型 ==================
enum EDataType : uint8_t
{
    DATA_TYPE_IR_IMAGE,            // 红外图像数据
    DATA_TYPE_RGB_IMAGE,           // 可见光图像数据
    DATA_TYPE_RADAR,               // 毫米波数据
    DATA_TYPE_TIME_MATCHED,        // 时间匹配后的多源数据
    DATA_TYPE_MMALG_RESULT,        // 多模态感知算法结果
    DATA_TYPE_POSEALG_RESULT,      // 姿态估计算法结果
    DATA_TYPE_LOCALG_RESULT,       // 目标定位算法结果
    DATA_TYPE_FUSION_RESULT,       // 融合结果（如有）
    DATA_TYPE_MAX
};

#endif
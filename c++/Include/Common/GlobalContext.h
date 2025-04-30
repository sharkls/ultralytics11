#ifndef _GLOBAL_CONTEXT__H_
#define _GLOBAL_CONTEXT__H_

#include <stdint.h>
#include <string>

enum TIMESTAMP_TYPE : uint8_t
{
    // 以下时间戳为软件部分赋值
    TIMESTAMP_MATCH,      // 用于时间匹配的时间戳
    TIMESTAMP_ARRIVE,     // 接收到原始数据的到机时间戳
    TIMESTAMP_MATCH_RECV, // 数据进入时间匹配之前的时间戳
    TIMESTAMP_MATCH_PUB,  // 时间匹配完成后发送数据的时间戳
    TIMESTAMP_MATCH_SUB,  // 订阅到时间匹配好的数据时的时间戳

    /*============点云相关时间戳类型=============*/
    // 以下时间戳为软件部分赋值
    TIMESTAMP_PC_FIRST_PACKET, // 雷达第0包点云pcap数据包内的时间戳，用于统计整套流程的总延时
    TIMESTAMP_PC_LAST_PACKET,  // 雷达第最后一包点云pcap数据包内的时间戳
    TIMESTAMP_PC_FRAME,        // 点云pcap数据包组成一帧完整点云时的本机时间戳，用于统计整套流程的总延时
    TIMESTAMP_PC_PUB,          // 发布单路点云时的时间戳
    TIMESTAMP_PC_SUB,          // 订阅到单路点云时的时间戳
    TIMESTAMP_PC_PACKET_PUB,   // 发布单路点云数据包结构时的时间戳
    TIMESTAMP_PC_PACKET_SUB,   // 订阅单路点云数据包结构时的时间戳
    TIMESTAMP_PC_RESULT_PUB,   // 发布点云识别结果时的时间戳
    TIMESTAMP_PC_RESULT_SUB,   // 订阅到点云识别结果时的时间戳

    // 以下时间戳为算法部分赋值
    TIMESTAMP_PC_PREALG_BEGIN,   // 进入点云预处理算法前时间戳
    TIMESTAMP_PC_PREALG_END,     // 点云预处理算法后时间戳
    TIMESTAMP_PC_DETALG_BEGIN,   // 进入点云检测算法前时间戳
    TIMESTAMP_PC_DETALG_END,     // 点云检测算法后时间戳
    TIMESTAMP_PC_TRACKALG_BEGIN, // 进入点云跟踪算法前时间戳
    TIMESTAMP_PC_TRACKALG_END,   // 点云跟踪算法后时间戳

    /*============点云信息时间戳类型=============*/
    // 以下时间戳为软件部分赋值
    TIMESTAMP_PCSRCINFO_SUB,
    // 对应各个相机的角度，从前往后，1、2、3、4
    // 基线版本使用此参数将点云与相机做时间匹配
    TIMESTAMP_PC_TIMEMATCH_CAMERA_1,
    TIMESTAMP_PC_TIMEMATCH_CAMERA_2,
    TIMESTAMP_PC_TIMEMATCH_CAMERA_3,
    TIMESTAMP_PC_TIMEMATCH_CAMERA_4,

    /*============视频相关时间戳类型============*/
    // 以下时间戳为软件部分赋值
    TIMESTAMP_VIDEO_RTP,        // 视频数据RTP时间戳
    TIMESTAMP_VIDEO_PUB,        // 发布视频数据时的时间戳
    TIMESTAMP_VIDEO_SUB,        // 订阅到视频数据时的时间戳
    TIMESTAMP_VIDEO_RESULT_PUB, // 发布视频识别结果时的时间戳
    TIMESTAMP_VIDEO_RESULT_SUB, // 订阅到视频识别结果时的时间戳
    TIMESTAMP_VIDEO_CAR_HEAD,   // 车头相机时间戳信息
    TIMESTAMP_VIDEO_CAR_BODY,   // 车身相机时间戳信息
    TIMESTAMP_VIDEO_CAR_TAIL,   // 车尾相机时间戳信息

    // 以下时间戳为算法部分赋值
    TIMESTAMP_VIDEO_PREALG_BEGIN,   // 进入视频预处理算法前时间戳
    TIMESTAMP_VIDEO_PREALG_END,     // 视频预处理算法后时间戳
    TIMESTAMP_VIDEO_DETALG_BEGIN,   // 进入视频检测算法前时间戳
    TIMESTAMP_VIDEO_DETALG_END,     // 视频检测算法后时间戳
    TIMESTAMP_VIDEO_TRACKALG_BEGIN, // 进入视频跟踪算法前时间戳
    TIMESTAMP_VIDEO_TRACKALG_END,   // 视频跟踪算法后时间戳

    /*============毫米波相关时间戳类型============*/
    // 以下时间戳为软件部分赋值
    TIMESTAMP_RADAR_TIME_MATCH, // 对应帧毫米波雷达的时间戳
    TIMESTAMP_RADAR_RESULT_PUB, // 发布毫米波结果时的时间戳
    TIMESTAMP_RADAR_RESULT_SUB, // 订阅到毫米波结果时的时间戳

    /*============融合相关时间戳类型============*/
    // 以下时间戳为软件部分赋值
    TIMESTAMP_FUSION_PC_RESULT_SUB,    // 订阅到点云识别结果数据时的时间戳
    TIMESTAMP_FUSION_VIDEO_RESULT_SUB, // 订阅到视频识别结果数据时的时间戳
    TIMESTAMP_FUSION_RADAR_RESULT_SUB, // 订阅到毫米波识别结果数据时的时间戳
    TIMESTAMP_FUSION_RESULT_PUB,       // 发布融合后的融合识别结果数据时的时间戳
    TIMESTAMP_FUSION_RESULT_SUB,       // 订阅到融合后的融合识别结果数据时的时间戳

    // 以下时间戳为算法部分赋值
    TIMESTAMP_FUSION_ALG_BEGIN, // 融合算法开始前时间戳
    TIMESTAMP_FUSION_ALG_END,   // 融合算法后时间戳

    /*============触发算法相关时间戳类型============*/
    // 以下部分为算法部分赋值
    TIMESTAMP_TRIGALG_BEGIN, // 触发算发开始时间戳
    TIMESTAMP_TRIGALG_END,   // 触发算发开始时间戳

    /*============跟踪算法相关时间戳类型============*/
    // 以下部分为算法部分赋值
    TIMESTAMP_TRACKALG_BEGIN, // 跟踪算法开始时间戳
    TIMESTAMP_TRACKALG_END,   // 跟踪算法开始时间戳

    TIMESTAMP_TYPE_MAX // 时间戳类型总个数
};

enum EFPSType : uint8_t
{
    // 以下帧率皆为软件部分赋值
    FPS_PC_INPUT_READ,    // 点云帧的输入帧率，表示从雷达输入到读取活动的帧率
    FPS_PC_OUTPUT_READ,   // 点云帧的输出帧率，表示从点云读取活动输出点云数据的帧率
    FPS_PC_INPUT_MATCH,   // 点云帧的输入帧率，表示点云数据输入到时间匹配活动的帧率
    FPS_PC_OUTPUT_MATCH,  // 时间匹配好的点云帧的输出帧率，表示从时间匹配活动输出匹配好的点云数据的帧率
    FPS_PC_MATCH_INPUT,   // 时间匹配好的点云帧的输入帧率，表示匹配好的点云数据输入到其他活动的帧率
    FPS_PC_RESULT_OUTPUT, // 点云结果的输出帧率，表示从点云算法活动输出点云结果数据的帧率
    FPS_PC_RESULT_INPUT,  // 点云结果的输入帧率，表示从点云结果数据输入到其他活动的帧率

    FPS_VIDEO_INPUT_READ,    // 视频帧的输入帧率，表示从相机输入到读取活动的帧率
    FPS_VIDEO_OUTPUT_READ,   // 视频帧的输出帧率，表示从视频读取活动输出视频数据的帧率
    FPS_VIDEO_INPUT_MATCH,   // 视频帧的输入帧率，表示从视频数据输入到时间匹配活动的帧率
    FPS_VIDEO_OUTPUT_MATCH,  // 时间匹配好的视频帧的输出帧率，表示从时间匹配活动输出匹配好的视频数据的帧率
    FPS_VIDEO_MATCH_INPUT,   // 时间匹配好的视频帧的输入帧率，表示从视频数据输入到其他活动的帧率
    FPS_VIDEO_RESULT_OUTPUT, // 视频结果的输出帧率，表示从视频算法活动输出视频结果数据的帧率
    FPS_VIDEO_RESULT_INPUT,  // 视频结果的输入帧率，表示从视频结果数据输入到其他活动的帧率

    FPS_RADAR_INPUT_READ,   // 毫米波帧的输入帧率，表示从毫米波雷达输入到读取活动的帧率
    FPS_RADAR_OUTPUT_READ,  // 毫米波帧的输出帧率，表示从毫米波读取活动输出毫米波数据的帧率
    FPS_RADAR_INPUT_MATCH,  // 毫米波帧的输入帧率，表示从毫米波数据输入到时间匹配活动的帧率
    FPS_RADAR_OUTPUT_MATCH, // 时间匹配好的毫米波帧的输出帧率，表示从时间匹配活动输出匹配好的毫米波数据的帧率
    FPS_RADAR_MATCH_INPUT,  // 时间匹配好的毫米波帧的输入帧率，表示从毫米波数据输入到其他活动的帧率

    FPS_FUSION_RESULT_OUTPUT, // 融合后的输出帧率，表示融合结果数据从融合算法活动输出的帧率
    FPS_FUSION_RESULT_INPUT,  // 融合的输入帧率，表示融合结果数据输入其他活动的帧率

    FPS_TRACK_RESULT_OUTPUT, // 跟踪后的输出帧率，表示跟踪结果数据从跟踪算法活动输出的帧率
    FPS_TRACK_RESULT_INPUT,  // 跟踪的输入帧率，表示跟踪结果数据输入其他活动的帧率
    FPS_TYPE_MAX             // FPS类型总个数
};

enum EDelayType : uint8_t
{
    // 以下部分为软件填充
    DELAY_TYPE_TIME_MATCH, // 时间匹配耗时

    DELAY_TYPE_PC_READ_ALL, // 表示从第0包点云数据时间开始，到点云数据从点云读取活动推送出去的总延时

    // 以下部分为点云算法填充
    DELAY_TYPE_PC_PREALG,        // 点云预处理耗时
    DELAY_TYPE_PC_DETECTION_ALG, // 时间匹配好的点云检测算法耗时
    DELAY_TYPE_PC_TRACK_ALG,     // 时间匹配好的点云跟踪算法耗时
    DELAY_TYPE_PC_ALG_ALL,       // 点云算法总耗时

    // 以下部分为软件填充
    DELAY_TYPE_PC_ALL, // 表示从时间匹配好的第0包点云数据时间开始，到点云结果数据从点云算法活动推送出去的总延时

    DELAY_TYPE_VIDEO_READ_ALL, // 表示从该帧视频数据起始时间开始，到该帧视频数据从视频读取活动推送出去的总延时

    // 以下部分为视频算法填充
    DELAY_TYPE_VIDEO_PREALG,        // 视频预处理耗时
    DELAY_TYPE_VIDEO_DETECTION_ALG, // 时间匹配好的视频检测算法耗时
    DELAY_TYPE_VIDEO_TRACK_ALG,     // 时间匹配好的视频跟踪算法耗时
    DELAY_TYPE_VIDEO_ALG_ALL,       // 视频算法总耗时

    // 以下部分为软件填充
    DELAY_TYPE_VIDEO_ALL, // 表示从该帧时间匹配好的视频数据起始时间开始，到该帧视频结果数据从视频算法活动推送出去的总延时

    DELAY_TYPE_RADAR_READ_ALL, // 表示从该帧毫米波数据起始时间开始，到该帧毫米波数据从毫米波读取活动推送出去的总延时

    // 以下部分为融合算法填充
    DELAY_TYPE_FUSION_ALG, // 表示融合算法总耗时
    DELAY_TYPE_FUSION_ALL, // 表示从匹配好的原始数据的最早时间戳开始，到融合后的结果数据从融合算法活动推送出去的总延时，即整套流程下来的总延时

    // 以下部分为场景算法填充
    DELAY_TYPE_SCENE_ALG,  // 场景算法总耗时

    DELAY_TYPE_TRIGGER_ALG, // 表示触发算法总耗时

    // 以下部分为跟踪算法填充
    DELAY_TYPE_TRACK_ALG, // 表示跟踪算法总耗时

    DELAY_TYPE_MAX // 延时类型总个数
};

enum DATA_SOURCE_TYPE : uint8_t
{
    // 以下数据来源类型在软件部分赋值
    DATA_SOURCE_ONLINE,  // 在线
    DATA_SOURCE_OFFLINE, // 离线
    DATA_SOURCE_TYPE_MAX // 数据体来源类型总个数
};

enum EDataType : uint8_t
{
    // 以下数据类型在软件部分赋值
    DATA_TYPE_PC = 0,        // 点云数据结构
    DATA_TYPE_PC_PACKET,     // 点云原始数据包结构
    DATA_TYPE_PC_TIME_MATCH, // 时间匹配好的点云数据结构
    DATA_TYPE_PC_RESULT,     // 点云结果数据结构

    DATA_TYPE_VIDEO,            // 视频数据结构
    DATA_TYPE_VIDEO_TIME_MATCH, // 时间匹配好的视频数据结构
    DATA_TYPE_VIDEO_RESULT,     // 视频结果数据结构

    DATA_TYPE_RADAR_RESULT,     // 毫米波结果数据结构
    DATA_TYPE_RADAR_TIME_MATCH, // 时间匹配好的毫米波数据结构
    DATA_TYPE_FUSION_RESULT,    // 融合结果数据结构
    DATA_TYPE_TRACK_RESULT,     // 跟踪结果数据结构

    // 以下数据类型在算法部分赋值
    DATA_TYPE_PC_VIDEO_FUSION_RESULT,       // 激光视频融合结果
    DATA_TYPE_PC_RADAR_FUSION_RESULT,       // 激光毫米波融合结果
    DATA_TYPE_VIDEO_RADAR_FUSION_RESULT,    // 视频毫米波融合结果
    DATA_TYPE_PC_VIDEO_RADAR_FUSION_RESULT, // 激光视频毫米波融合结果

    DATA_TYPE_V2X,                           //  場景v2x结果
    DATA_TYPE_TWIN_DISPLAY,                  //  場景twin结果

    DATA_TYPE_MAX // 数据体类型总个数
};

namespace isfp_viewer
{
    const std::string kGetParamAuthorization = "getParam_Authorization";
    const std::string kSetParamAuthorization = "setParam_Authorization";
    const std::string kGetParamCamera = "getParam_camera";
    const std::string kSetParamCamera = "setParam_camera";
    const std::string kGetParamLidar = "getParam_lidar";
    const std::string kSetParamLidar = "setParam_lidar";
    const std::string kGetParamNet = "getParam_forwardNet";
    const std::string kSetParamNet = "setParam_forwardNet";
    const std::string kGetParamPcAlg = "getParam_pcAlg";
    const std::string kSetParamPcAlg = "setParam_pcAlg";
    const std::string kGetParamStartControl = "getParam_startControl";
    const std::string kSetParamStartControl = "setParam_startControl";
    const std::string kGetParamStation = "getParam_station";
    const std::string kSetParamStation = "setParam_station";
};

#endif
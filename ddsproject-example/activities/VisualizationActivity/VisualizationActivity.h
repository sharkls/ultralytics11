#ifndef VisualizationACTIVITY_H
#define VisualizationACTIVITY_H

#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

// #include "activities/alg_activity/proto/alg_activity.pb.h"
#include "activities/VisualizationActivity/proto/VisualizationActivity.pb.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcData.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcDataPubSubTypes.h"
#include "activities/idl/CAlgResult/CAlgResult.h"
#include "activities/idl/CAlgResult/CAlgResultPubSubTypes.h"
#include "include/queue/CSafeDataDeque.h"
#include "include/Interface/CSelfAlgParam.h"
#include "include/Common/Functions.h"
#include <fstream>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <atomic>
#include <mutex>

// CounterTopic是话题，test0_acticity向test1_activity发送数据
class VisualizationActivity : public ActivityBase
{
public:
    VisualizationActivity();
    ~VisualizationActivity();

    // // 软件处理算法结果的回调函数
    // void GetVisualizationResultResponseMessageCallbackFunc(const CAlgResult& res_message, void* data_handle);

protected:
    // 初始化，读取配置文件
    virtual bool Init() override;
    // 当收到master节点的RUN指令，则执行Start，用于启动线程
    virtual void Start() override;
    // 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
    virtual void PauseClear();

private:
    // 向 Multi_Modal_Fusion_Result_Topic 中发送消息
    void MessageProducerThreadFunc();
    // 处理camera_merged_topic中的数据
    void MessageConsumerThreadFunc();
    
    // 读取目标定位结果
    void ReadObjectLocationResultCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name);
    // 读取多模态原始数据结果
    void ReadMultiModalSrcDataCallbackFunc(const CMultiModalSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name);

    // socket
    void openSocket();
    void closeSocket();

private:
    // DDS 读取和写出节点
    std::shared_ptr<Reader<CMultiModalSrcData>> reader_multi_modal_src_data_;
    std::shared_ptr<Reader<CAlgResult>> reader_object_location_result_;
    std::shared_ptr<Writer> writer_visualization_result_;

    // 消息队列
    CSafeDataDeque<std::shared_ptr<CMultiModalSrcData>> multi_modal_src_data_deque_;   // 时间同步后的数据
    CSafeDataDeque<std::shared_ptr<CAlgResult>> object_location_result_deque_;  // 目标定位结果


    // 消息发送和消费线程
    std::unique_ptr<std::thread> message_producer_thread_;
    std::unique_ptr<std::thread> message_consumer_thread_;
    std::atomic<bool> is_running_{false};
    std::mutex thread_mutex_;

    // socket
    int serv_sock{-1};
    int clnt_sock{-1};
};

#endif // VisualizationACTIVITY_H
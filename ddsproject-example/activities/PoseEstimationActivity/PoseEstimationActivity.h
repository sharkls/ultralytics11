#ifndef POSEESTIMATIONACTIVITY_H
#define POSEESTIMATIONACTIVITY_H

#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

// #include "activities/alg_activity/proto/alg_activity.pb.h"
#include "activities/PoseEstimationActivity/proto/PoseEstimationActivity.pb.h"
#include "activities/idl/CAlgResult/CAlgResult.h"
#include "activities/idl/CAlgResult/CAlgResultPubSubTypes.h"
#include "activities/idl/CAlgResult/CAlgResult.h"
#include "activities/idl/CAlgResult/CAlgResultPubSubTypes.h"
#include "include/queue/CSafeDataDeque.h"
#include "include/Interface/ExportPoseEstimationAlgLib.h"
#include "include/Interface/CSelfAlgParam.h"
#include "include/Common/Functions.h"
#include "include/Common/FunctionHub.h"
#include <fstream>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <atomic>
#include <mutex>

// CounterTopic是话题，test0_acticity向test1_activity发送数据
class PoseEstimationActivity : public ActivityBase
{
public:
    PoseEstimationActivity();
    ~PoseEstimationActivity();

    // 软件处理算法结果的回调函数
    void GetPoseEstimationResultResponseMessageCallbackFunc(const CAlgResult& res_message, void* data_handle);

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
    
    void ReadCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name);


private:
    // DDS 读取和写出节点
    std::shared_ptr<Reader<CAlgResult>> reader_;
    std::shared_ptr<Writer> writer_;

    // 消息队列
    CSafeDataDeque<std::shared_ptr<CAlgResult>> camera_merged_data_deque_;   // 时间同步后的数据
    CSafeDataDeque<std::shared_ptr<CAlgResult>> pose_estimation_result_deque_;  // 多模态融合结果

    // 消息发送和消费线程
    std::unique_ptr<std::thread> message_producer_thread_;
    std::unique_ptr<std::thread> message_consumer_thread_;
    std::atomic<bool> is_running_{false};
    std::mutex thread_mutex_;

    // 算法实例
    IPoseEstimationAlg *pose_estimation_alg_{nullptr};
    std::string root_path_;
    CSelfAlgParam alg_param_;

    // 测试
    int64_t endTimeStamp_{0};
    int64_t startTimeStamp_{0};
    int32_t count_{0};
    int64_t count_time_{0};
};

#endif // POSEESTIMATIONACTIVITY_H
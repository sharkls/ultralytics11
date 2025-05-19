#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

#include "activities/alg_activity/proto/alg_activity.pb.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcData.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcDataPubSubTypes.h"

#include <fstream>

#include <opencv2/opencv.hpp>

// CounterTopic是话题，test0_acticity向test1_activity发送数据

class ActivityAlg : public ActivityBase
{
public:
    ActivityAlg();
    ~ActivityAlg();

protected:
    // 初始化，读取配置文件
    virtual bool Init() override;
    // 当收到master节点的RUN指令，则执行Start，用于启动线程
    virtual void Start() override;
    // 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
    virtual void PauseClear();

private:
    // 向test1_activity发送数据的线程，向CounterTopic中发送消息
    void CounterMessageProducerThreadFunc();
    
    void ReadCallbackFunc(const CMultiModalSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name);

private:
    std::shared_ptr<Reader<CMultiModalSrcData>> reader_;
    std::shared_ptr<Writer> writer_;

    std::thread *counter_message_producer_thread_{nullptr};

    void saveToDisk_XYZ(const int serial, const std::vector<uint8_t>& img);
    void saveToDisk_M3JUVC(const int serial, const std::vector<uint8_t>& img);
};

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(ActivityAlg)

ActivityAlg::ActivityAlg()
{
}

ActivityAlg::~ActivityAlg()
{
    if ((counter_message_producer_thread_ != nullptr) && (counter_message_producer_thread_->joinable()))
    {
        counter_message_producer_thread_->join();
        delete counter_message_producer_thread_;
        counter_message_producer_thread_ = nullptr;
    }
}

bool ActivityAlg::Init()
{
    // 读取私有配置文件内容
    TopicConfig topic_config;
    TINFO << config_file_path_;
    if (!GetProtoConfig(&topic_config))
    {
        TINFO << "configure file parse failed";
        return false;
    }


    eprosima::fastdds::dds::TypeSupport deal_data_type(new CMultiModalSrcDataPubSubType());
    if (!node_->AddTopic(topic_config.camera_merged_topic(), deal_data_type))
    {
        return false;
    }
    // 创建CounterTopic对应的reader，接收数据
    reader_ = node_->CreateReader<CMultiModalSrcData>(topic_config.camera_merged_topic(),
        std::bind(&ActivityAlg::ReadCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_->Init();


    // 添加CounterTopic
    // eprosima::fastdds::dds::TypeSupport deal_data_type(new CounterMessagePubSubType());
    // if (!node_->AddTopic(topic_config.counter_topic(), deal_data_type))
    // {
    //     return false;
    // }
    // // 创建CounterTopic对应的writer，发送数据
    // writer_ = node_->CreateWriter(topic_config.counter_topic());
    // // 初始化
    // writer_->Init();

    return true;
}

void ActivityAlg::ReadCallbackFunc(const CMultiModalSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name)
{
    std::shared_ptr<CMultiModalSrcData> message_ptr = std::make_shared<CMultiModalSrcData>(message);

    std::vector<CVideoSrcData> vecCVideoSrcData = message_ptr->vecVideoSrcData();

    std::cout << "=====: " << vecCVideoSrcData[1].unFrameId() << " " <<  vecCVideoSrcData[1].ulTimestampPub() << std::endl;

    // 2764800 cv::Mat matDepth(cv::Size(1280, 720), CV_8UC3, p);
    saveToDisk_XYZ(vecCVideoSrcData[0].unFrameId(), vecCVideoSrcData[0].vecImageBuf());

    // 327680 cv::Mat matDepth(cv::Size(640, 512), CV_8UC1, p);
    saveToDisk_M3JUVC(vecCVideoSrcData[1].unFrameId(), vecCVideoSrcData[1].vecImageBuf());
}

void ActivityAlg::saveToDisk_M3JUVC(const int serial, const std::vector<uint8_t>& img)
{
    // 640 * 512 == 327680
    void *p = malloc(640 * 512);
    memcpy(p, img.data(), 640 * 512);
    cv::Mat matImg(cv::Size(640, 512), CV_8UC1, p);
    std::string filePath = "/workspace/ddsproject-example/tmp/M3JUVC/" + std::to_string(serial) + ".jpg";
    if (cv::imwrite(filePath, matImg)) {
        ;
    } else {

    }

    free(p);
}

void ActivityAlg::saveToDisk_XYZ(const int serial, const std::vector<uint8_t>& img)
{
    // 1280 * 720 * 3 == 2764800
    void *p = malloc(1280 * 720 * 3);
    memcpy(p, img.data(), 1280 * 720 * 3);
    cv::Mat matImg(cv::Size(1280, 720), CV_8UC3, p);
    std::string filePath = "/workspace/ddsproject-example/tmp/xyz/" + std::to_string(serial) + ".jpg";
    if (cv::imwrite(filePath, matImg)) {
        ;
    } else {

    }

    free(p);
}

void ActivityAlg::Start()
{
    TINFO << "ActivityAlg running";
    counter_message_producer_thread_ = new std::thread(&ActivityAlg::CounterMessageProducerThreadFunc, this);
}

void ActivityAlg::PauseClear()
{
    if ((counter_message_producer_thread_ != nullptr) && (counter_message_producer_thread_->joinable()))
    {
        counter_message_producer_thread_->join();
        delete counter_message_producer_thread_;
        counter_message_producer_thread_ = nullptr;
    }
}

void ActivityAlg::CounterMessageProducerThreadFunc()
{
    // 向CounterTopic中循环发送数据
    // is_running_是线程结束的标志位，通过master的指令进行控制
    while (is_running_.load())
    {
        // CounterMessage message;
        // message.cnt(cnt_);
        // message.tip("message wait for data");
        // writer_->SendMessage((void *)&message);
        // cnt_++;
        // TINFO << "-----------------------------------------------" << cnt_;
        // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "../../../ddsproject-example/activities/conf/alg_activity.info";
    ActivityInfo activity_info;
    // 解析test0_activity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("ActivityAlg");

    ActivityAlg *activity = new ActivityAlg();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}

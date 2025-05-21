#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

#include "activities/test0_activity/proto/test0_activity.pb.h"
#include "activities/idl/counter_message/counter_message.h"
#include "activities/idl/counter_message/counter_messagePubSubTypes.h"
#include "activities/idl/counter_response_message/counter_response_message.h"
#include "activities/idl/counter_response_message/counter_response_messagePubSubTypes.h"

// CounterTopic是话题，test0_acticity向test1_activity发送数据

class ActivityTest0 : public ActivityBase
{
public:
    ActivityTest0();
    ~ActivityTest0();

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

private:
    std::shared_ptr<Writer> writer_;

    std::thread *counter_message_producer_thread_{nullptr};

    int cnt_{0};
};

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(ActivityTest0)

ActivityTest0::ActivityTest0()
{
}

ActivityTest0::~ActivityTest0()
{
    if ((counter_message_producer_thread_ != nullptr) && (counter_message_producer_thread_->joinable()))
    {
        counter_message_producer_thread_->join();
        delete counter_message_producer_thread_;
        counter_message_producer_thread_ = nullptr;
    }
}

bool ActivityTest0::Init()
{
    // 读取私有配置文件内容
    TopicConfig topic_config;
    TINFO << config_file_path_;
    if (!GetProtoConfig(&topic_config))
    {
        TINFO << "configure file parse failed";
        return false;
    }

    // 添加CounterTopic
    eprosima::fastdds::dds::TypeSupport deal_data_type(new CounterMessagePubSubType());
    if (!node_->AddTopic(topic_config.counter_topic(), deal_data_type))
    {
        return false;
    }
    // 创建CounterTopic对应的writer，发送数据
    writer_ = node_->CreateWriter(topic_config.counter_topic());
    // 初始化
    writer_->Init();

    return true;
}

void ActivityTest0::Start()
{
    TINFO << "ActivityTest0 running";
    counter_message_producer_thread_ = new std::thread(&ActivityTest0::CounterMessageProducerThreadFunc, this);
}

void ActivityTest0::PauseClear()
{
    if ((counter_message_producer_thread_ != nullptr) && (counter_message_producer_thread_->joinable()))
    {
        counter_message_producer_thread_->join();
        delete counter_message_producer_thread_;
        counter_message_producer_thread_ = nullptr;
    }
}

void ActivityTest0::CounterMessageProducerThreadFunc()
{
    // 向CounterTopic中循环发送数据
    // is_running_是线程结束的标志位，通过master的指令进行控制
    while (is_running_.load())
    {
        CounterMessage message;
        message.cnt(cnt_);
        message.tip("message wait for data");
        writer_->SendMessage((void *)&message);
        cnt_++;
        TINFO << "-----------------------------------------------" << cnt_;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "../../../ddsproject-example/activities/conf/test0_activity.info";
    ActivityInfo activity_info;
    // 解析test0_activity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("ActivityTest0");

    ActivityTest0 *activity = new ActivityTest0();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}

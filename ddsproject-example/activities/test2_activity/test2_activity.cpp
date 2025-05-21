#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

#include "activities/test2_activity/proto/test2_activity.pb.h"
#include "activities/idl/counter_response_message/counter_response_message.h"
#include "activities/idl/counter_response_message/counter_response_messagePubSubTypes.h"

class ActivityTest2 : public ActivityBase
{
public:
    ActivityTest2();
    ~ActivityTest2();

protected:
    // 初始化，读取配置文件、其他成员的初始化操作
    virtual bool Init() override;
    // 当收到master节点的RUN指令，则执行Start，用于启动线程
    virtual void Start() override;

private:
    // 从CounterResponseTopic中获取CounterMessage的回调函数，将CounterMessage存储到消息队列中
    void ReadCounterResponseMessageCallbackFunc(const CounterResponseMessage &message, void *data_handle, std::string node_name, std::string topic_name);

private:
    std::shared_ptr<Reader<CounterResponseMessage>> reader_;
};

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(ActivityTest2)

ActivityTest2::ActivityTest2()
{
}

ActivityTest2::~ActivityTest2()
{
}

bool ActivityTest2::Init()
{
    // 读取私有配置文件内容
    TopicConfig topic_config;
    if (!GetProtoConfig(&topic_config))
    {
        TINFO << "configure file parse failed";
        return false;
    }

    // 添加CounterResponseTopic
    eprosima::fastdds::dds::TypeSupport deal_data_type(new CounterResponseMessagePubSubType());
    if (!node_->AddTopic(topic_config.counter_response_topic(), deal_data_type))
    {
        return false;
    }
    // 创建CounterResponseTopic对应的reader，接收数据
    reader_ = node_->CreateReader<CounterResponseMessage>(topic_config.counter_response_topic(), std::bind(&ActivityTest2::ReadCounterResponseMessageCallbackFunc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    // 初始化
    reader_->Init();

    return true;
}

void ActivityTest2::Start()
{
    TINFO << "ActivityTest2 running";
}

void ActivityTest2::ReadCounterResponseMessageCallbackFunc(const CounterResponseMessage &message, void *data_handle, std::string node_name, std::string topic_name)
{
    TINFO << "ActivityTest2 recv message data : [ " << message.cnt() << " ], data_name [ " << message.response() << " ]";
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "/workspace/ddsproject-example/activities/conf/test2_activity.info";
    ActivityInfo activity_info;
    // 解析test0_activity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);
    
    // 设置当前进程名，用于在日志中打印
    SetName("ActivityTest2");

    ActivityTest2* activity = new ActivityTest2();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}

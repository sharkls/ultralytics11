#ifndef ACTIVITY_ACTIVITY_BASE_HPP
#define ACTIVITY_ACTIVITY_BASE_HPP

#include "include/class_loader/class_loader_register_macro.hpp"
#include "include/common/file.hpp"
#include "include/common/environment.hpp"
#include "include/common/global_data.hpp"
#include "include/node/node.hpp"
#include "include/common/state.hpp"

#include "proto/activity_cfg/activity_conf.pb.h"
#include "proto/activity_cfg/master_activity_communicate_conf.pb.h"
#include "idl/master_activity_message/master_activity_messagePubSubTypes.h"
#include "idl/master_activity_message/master_activity_message.h"

class ActivityBase
{
public:
    ActivityBase();
    ~ActivityBase();

    bool Initialize(const ActivityInfo &activity_info);
    void Run();                  // 运行
    void Shutdown();             // 关闭

    std::string GetActivityClassName() const;

    template <typename T>
    bool GetProtoConfig(T *config) const
    {
        return GetProtoFromFile(config_file_path_, config);
    }

protected:
    virtual bool Init() = 0; // 其他初始化，创建writer或者reader
    virtual void Start() { return; }
    virtual void PauseClear() { return; }
    virtual void Clear() { return; }

protected:
    void LoadConfigFiles(const ActivityConfig &activity_config)
    {
        if (!activity_config.conf_file_path().empty())
        {
            if (activity_config.conf_file_path()[0] != '/')
            {
                config_file_path_ = GetAbsolutePath(WorkRoot(), activity_config.conf_file_path());
            }
            else
            {
                config_file_path_ = activity_config.conf_file_path();
            }
        }
        TINFO << "config_file_path: [ " << config_file_path_ << " ]";
    }

    void SendActivityStatusFeedback(const ActivityStatusEnum);

    std::string config_file_path_ = "";
    std::shared_ptr<Node> node_ = nullptr;
    std::atomic<bool> is_running_{false};

    std::shared_ptr<Reader<ActivityCmdControlMessage>> activity_cmd_control_reader_{nullptr};
    std::shared_ptr<Writer> activity_status_feedback_writer{nullptr};

private:
    void ActivityCmdControlMessageCallbackHandleFunc(const ActivityCmdControlMessage &message, void *data_handle, std::string node_name, std::string topic_name);

private:
    std::string activity_name_ = "";
    ActivityStatusFeedbackMessage activity_status_feedback_message_;

    std::atomic<ActivityRunCmdEnum> cmd_{NO_CMD};
    std::atomic<bool> is_shutdown_{false};
};

#define REGISTER_ACTIVITY(name) \
    CLASS_LOADER_REGISTER_CLASS(name, ActivityBase)

#endif
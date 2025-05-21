#ifndef ACTIVITY_MASTER_ACTIVITY_HPP
#define ACTIVITY_MASTER_ACTIVITY_HPP    

#include <vector>
#include <atomic>

#include "include/common/file.hpp"
#include "include/common/environment.hpp"
#include "include/common/log.hpp"
#include "include/node/node.hpp"
#include "include/common/global_data.hpp"
#include "include/common/state.hpp"
#include "include/common/macros.hpp"

#include "proto/activity_conf.pb.h"
#include "proto/master_activity_communicate_conf.pb.h"
#include "proto/activity_launch_conf.pb.h"
#include "idl/master_activity_message/master_activity_messagePubSubTypes.h"
#include "idl/master_activity_message/master_activity_message.h"

class MasterActivity
{
public:
    ~MasterActivity();

    bool Init();
    void Run();
    void Shutdown();

private:
    void Process();
    void Clear();

private:
    void ActivityStatusFeedbackMessageCallbackHandleFunc(const ActivityStatusFeedbackMessage &message, void *data_handle, std::string node_name, std::string topic_name);
    
    std::shared_ptr<Node> node_ = nullptr;
    std::shared_ptr<Reader<ActivityStatusFeedbackMessage>> activity_status_feedback_reader{nullptr};
    std::shared_ptr<Writer> activity_cmd_ctrl_writer{nullptr};

    std::vector<std::string> activity_list_;

    ModuleLaunchConfig module_launch_config_;
    ActivityCmdControlMessage activity_cmd_control_message_;

    ActivityInfo master_config_;

    std::atomic<bool> is_shutdown_{false};

    DECLARE_SINGLETON(MasterActivity)
};

#endif
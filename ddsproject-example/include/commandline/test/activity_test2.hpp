#ifndef COMMANDLINE_TEST_ACTIVITY_TEST2_HPP
#define COMMANDLINE_TEST_ACTIVITY_TEST2_HPP

#include "include/activity/base/activitybase.hpp"
#include "include/common/log.hpp"

#include "proto/activity_cfg/activity_conf.pb.h"
#include "output/test/commandline/test/idl/message_res_display/message_res_display.h"
#include "output/test/commandline/test/idl/message_res_display/message_res_displayPubSubTypes.h"

class ActivityTest2 : public ActivityBase
{
public:
    ActivityTest2();
    ~ActivityTest2();

protected:
    virtual bool Init() override;
    virtual void Start() override;

private:
    void ReadMessageResDisplayCallbackFunc(const MessageResDisplay &message, void *data_handle, std::string node_name, std::string topic_name);

private:
    std::shared_ptr<Reader<MessageResDisplay>> reader_;
};

REGISTER_ACTIVITY(ActivityTest2)
#endif
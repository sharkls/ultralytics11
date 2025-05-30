#ifndef COMMANDLINE_TEST_ACTIVITY_TEST0_HPP
#define COMMANDLINE_TEST_ACTIVITY_TEST0_HPP

#include <thread>

#include "include/activity/base/activitybase.hpp"
#include "include/common/binary.hpp"

#include "output/test/commandline/test/idl/message_wait_deal/message_wait_deal.h"
#include "output/test/commandline/test/idl/message_wait_deal/message_wait_dealPubSubTypes.h"

class ActivityTest0 : public ActivityBase
{
public:
    ActivityTest0();
    ~ActivityTest0();

protected:
    virtual bool Init() override;
    virtual void Start() override;

private:
    void DealDataCustomThread();

private:
    std::shared_ptr<Writer> writer_;

    std::thread* deal_data_custom_thread_;

    int cnt_{0};
};

REGISTER_ACTIVITY(ActivityTest0)

#endif
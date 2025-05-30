#ifndef COMMANDLINE_TEST_ACTIVITY_TEST3_HPP
#define COMMANDLINE_TEST_ACTIVITY_TEST3_HPP

#include <functional>
#include <thread>
#include <memory>

#include "include/activity/base/activitybase.hpp"
#include "include/common/log.hpp"
#include "include/queue/deque_queue.hpp"

#include "output/test/commandline/test/idl/message_wait_deal/message_wait_deal.h"
#include "output/test/commandline/test/idl/message_wait_deal/message_wait_dealPubSubTypes.h"
#include "output/test/commandline/test/idl/message_res_display/message_res_display.h"
#include "output/test/commandline/test/idl/message_res_display/message_res_displayPubSubTypes.h"

#include "include/commandline/test/test_obj.hpp"

// using ActivityTestCallback = std::function<void(,void*)>;

class ActivityTest3 : public ActivityBase
{
public:
    ActivityTest3();
    ~ActivityTest3();

protected:
    virtual bool Init() override;
    virtual void Start() override;

private:
    void ReadMessageWaitForDealDataCallbackFunc(const MessageWaitForDeal &message, void *data_handle, std::string node_name, std::string topic_name);
    void DealMessageDataThreadFunc();
    void SendMessageResDisplayThreadFunc();
    void GetResultMessageCallbackFunc(const MessageResDisplay& res_message, void* data_handle);

private:
    SafeDataQueue<MessageWaitForDeal>* message_wait_for_deal_queue_;
    std::thread* deal_data_customer_thread_{nullptr};

    SafeDataQueue<MessageResDisplay>* message_res_display_queue_;
    std::thread* res_data_customer_thread_{nullptr};

    std::shared_ptr<Reader<MessageWaitForDeal>> reader_;
    std::shared_ptr<Writer> writer_;

    TestObj* test_obj_;
};

REGISTER_ACTIVITY(ActivityTest3)
#endif
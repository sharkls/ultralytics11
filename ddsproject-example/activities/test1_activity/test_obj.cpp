#include "activities/test1_activity/test_obj.hpp"

TestObj::TestObj()
{
}

TestObj::~TestObj()
{
}

void TestObj::Init(const TestObjCallback &cb, void* data_handler)
{
    cb_ = cb;
    data_handler_ = data_handler;
}

void TestObj::Run(const std::shared_ptr<CounterMessage> &message)
{
    CounterResponseMessage res_message;
    res_message.cnt(message->cnt() + 1000);
    res_message.response("after deal message");
    cb_(res_message, data_handler_);
}
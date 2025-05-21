#ifndef ACTIVITIES_TEST1_ACTIVITY_TEST_OBJ_HPP
#define ACTIVITIES_TEST1_ACTIVITY_TEST_OBJ_HPP

#include <functional>

#include "activities/idl/counter_message/counter_message.h"
#include "activities/idl/counter_response_message/counter_response_message.h"

using TestObjCallback = std::function<void(const CounterResponseMessage&, void*)>;

class TestObj
{
public:
    TestObj();
    ~TestObj();

    void Init(const TestObjCallback& cb, void* data_handler);
    void Run(const std::shared_ptr<CounterMessage>& message);

protected:


private:
    TestObjCallback cb_;
    void* data_handler_;
};

#endif
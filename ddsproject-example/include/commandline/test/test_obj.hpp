#ifndef COMMANDLINE_TEST_TEST_OBJ_HPP
#define COMMANDLINE_TEST_TEST_OBJ_HPP

#include <functional>

#include "output/test/commandline/test/idl/message_wait_deal/message_wait_deal.h"
#include "output/test/commandline/test/idl/message_res_display/message_res_display.h"

using TestObjCallback = std::function<void(const MessageResDisplay&, void*)>;

class TestObj
{
public:
    TestObj();
    ~TestObj();

    void Init(const TestObjCallback& cb, void* data_handler);
    void Run(const std::shared_ptr<MessageWaitForDeal>& message);

protected:


private:
    TestObjCallback cb_;
    void* data_handler_;
};

#endif
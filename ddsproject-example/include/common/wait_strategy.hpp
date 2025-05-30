#ifndef COMMON_WAIT_STRATEGY_HPP
#define COMMON_WAIT_STRATEGY_HPP

#include <cstdlib>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <mutex>

class WaitStrategy
{
public:
    virtual ~WaitStrategy() {};
    virtual void NotifyOne() {};
    virtual void NotifyAll() {};
    virtual bool EmptyWait() = 0;
};

/**
 * @brief 休眠等待：使用线程休眠函数
*/
class SleepWaitStrategy : public WaitStrategy
{
public:
    SleepWaitStrategy(){};
    explicit SleepWaitStrategy(uint64_t sleep_time_us) : sleep_time_us_(sleep_time_us){}

    bool EmptyWait()
    {
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us_));
        return true;
    }

    void SetSleepTimeMicroSeconds(uint64_t sleep_time_us)
    {
        sleep_time_us_ = sleep_time_us;
    }

private:
    uint64_t sleep_time_us_{10000};
};

/**
 * 阻塞等待
*/
class BlockWaitStrategy : public WaitStrategy
{
public:
    BlockWaitStrategy() {};

    void NotifyOne() override
    {
        cv_.notify_one();
    }

    void NotifyAll() override
    {
        cv_.notify_all();
    }

    bool EmptyWait()
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock); // 等待直到满足条件
        return true;
    } 

private:
    std::condition_variable cv_;
    std::mutex mtx_;
};

/**
 * 超时等待
*/
class TimeoutBlockWaitStrategy : public WaitStrategy
{
public:
    TimeoutBlockWaitStrategy(){}
    explicit TimeoutBlockWaitStrategy(uint64_t timeout) : timeout_(std::chrono::milliseconds(timeout)) {}

    void NotifyOne() override
    {
        cv_.notify_one();
    }
    
    void NotifyAll() override
    {
        cv_.notify_all();
    }

    bool EmptyWait() override
    {
        std::unique_lock<std::mutex> lock(mtx_);
        if (cv_.wait_for(lock, timeout_) == std::cv_status::timeout) { // 等待timeout时间，超时返回
            return false;
        }
        return true;
    }

    void setTimeout(uint64_t timeout)
    {
        timeout_ = std::chrono::milliseconds(timeout);
    }

private:
    std::condition_variable cv_;
    std::mutex mtx_;
    std::chrono::milliseconds timeout_;
};

/**
 * 让出当前的时间片
*/
class YieldWaitStrategy : public WaitStrategy
{
public:
    YieldWaitStrategy() {}

    bool EmptyWait() override
    {
        std::this_thread::yield();
        return true;
    }
};

/**
 * 忙等待，不停的询问
*/
class BusyWaitStrategy : public WaitStrategy
{
public:
    BusyWaitStrategy() {}
    
    bool EmptyWait() override
    {
        return true;
    }
};

#endif
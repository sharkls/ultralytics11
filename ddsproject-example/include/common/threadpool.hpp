#ifndef COMMON_THREADPOOL_HPP
#define COMMON_THREADPOOL_HPP

#include <iostream>
#include <memory>
#include <future>
#include <thread>
#include <queue>
#include <functional>
#include <atomic>
#include <condition_variable>

#include "include/common/bound_queue.hpp"

class ThreadPool
{
public:
    explicit ThreadPool(std::size_t thread_num, std::size_t max_task_num = 1000);
    ~ThreadPool();

    template <typename F, typename... Args>
    auto Enqueue(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> workers_;
    BoundedQueue<std::function<void()>> task_queue_;
    std::atomic<bool> quit_;

    std::condition_variable cv_;
    std::mutex mtx_;
};

ThreadPool::ThreadPool(std::size_t thread_num, std::size_t max_task_num)
    : quit_(false)
{
    if (!task_queue_.Init(max_task_num, new TimeoutBlockWaitStrategy()))
    {
        throw std::runtime_error("Task queue init failed!");
    }
    workers_.reserve(thread_num);
    for (size_t i = 0; i < thread_num; i++)
    {
        workers_.emplace_back([this]
                              {
            while (!quit_)
            {
                std::function<void()> task;
                if (this->task_queue_.WaitDequeue(&task))
                {
                    task();
                }
            } });
    }
}

ThreadPool::~ThreadPool()
{
    if (quit_.exchange(true))
    {
        return;
    }

    task_queue_.BreakAllWait();

    for (std::thread &worker : workers_)
    {
        worker.join();
    }
}

template <typename F, typename... Args>
auto ThreadPool::Enqueue(F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using result_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<result_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<result_type> res = task->get_future();
    if (quit_)
    {
        return std::future<result_type>();
    }

    task_queue_.Enqueue([task](){
        (*task)();
    });
    
    return res;
}

#endif
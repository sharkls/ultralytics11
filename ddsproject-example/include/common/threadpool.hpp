#ifndef COMMON_THREADPOOL_HPP
#define COMMON_THREADPOOL_HPP

#include <memory>
#include <future>
#include <thread>
#include <queue>
#include <functional>
#include <atomic>

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
    std::queue<std::function<void()>> task_queue_;
    std::atomic<bool> quit_;
};

ThreadPool::ThreadPool(std::size_t thread_num, std::size_t max_task_num)
{
    workers_.reserve(thread_num);
    for (size_t i; i < thread_num; i++)
    {
        workers_.emplace_back([this]
                              {
            while ( !quit_)
            {
                std::function<void()> task;
                if (true)   // 从队列中获取数据
                {
                    task();
                }
            } });
    }
}

ThreadPool::~ThreadPool()
{
}

#endif

template <typename F, typename... Args>
auto ThreadPool::Enqueue(F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using result_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<result_type>(std::bind())>

    return std::future<typename std::result_of<F(Args...)>::type>();
}

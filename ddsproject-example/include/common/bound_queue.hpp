#ifndef COMMON_BOUNDED_QUEUE_HPP
#define COMMON_BOUNDED_QUEUE_HPP

#include <iostream>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <utility>

#include "include/common/wait_strategy.hpp"

template <typename T>
class BoundedQueue
{
public:
    BoundedQueue() {};
    ~BoundedQueue();

    BoundedQueue &operator=(const BoundedQueue &bounded_queue) = delete;
    BoundedQueue(const BoundedQueue &bounded_queue) = delete;

    bool Init(uint64_t size);
    bool Init(uint64_t size, WaitStrategy *wait_strategy);
    bool Enqueue(const T &element);
    bool Enqueue(T &&element);
    bool Dequeue(T *element);

    bool WaitEnqueue(const T &element);
    bool WaitEnqueue(T &&element);
    bool WaitDequeue(T *element);

    uint64_t Size();
    bool Empty();
    uint64_t Head();
    uint64_t Tail();
    uint64_t Commit();

    void SetWaitStrategy(WaitStrategy *wait_strategy);
    void BreakAllWait(); // notify_all

private:
    uint64_t GetIndex(uint64_t num);

    uint64_t pool_size_{0};
    T *pool_ = nullptr;

    std::atomic<uint64_t> head_{0};
    std::atomic<uint64_t> tail_{1};
    std::atomic<uint64_t> commit_{1};

    std::unique_ptr<WaitStrategy> wait_strategy_{nullptr};

    volatile bool notify_all_{false};
};

template <typename T>
BoundedQueue<T>::~BoundedQueue()
{
    if (wait_strategy_)
    {
        BreakAllWait();
    }
    if (pool_)
    {
        for (uint64_t i = 0; i < pool_size_; ++i)
        {
            pool_[i].~T();
        }
        std::free(pool_);
    }
}

template <typename T>
bool BoundedQueue<T>::Init(uint64_t size)
{
    return Init(size, new SleepWaitStrategy());
}

template <typename T>
bool BoundedQueue<T>::Init(uint64_t size, WaitStrategy *wait_strategy)
{
    pool_size_ = size + 2;
    pool_ = reinterpret_cast<T *>(std::calloc(pool_size_, sizeof(T)));
    if (pool_ == nullptr)
    {
        return false;
    }
    for (uint64_t i = 0; i < pool_size_; i++)
    {
        new (&(pool_[i])) T(); // placement new
    }
    wait_strategy_.reset(wait_strategy);
    return true;
}

template <typename T>
bool BoundedQueue<T>::Enqueue(const T &element)
{
    uint64_t new_tail = 0;
    uint64_t old_commit = 0;
    uint64_t old_tail = tail_.load(std::memory_order_acquire);

    do
    {
        new_tail = old_tail + 1;
        if (GetIndex(new_tail) == GetIndex(head_.load(std::memory_order_acquire)))
        {
            return false;
        }
    } while (!tail_.compare_exchange_weak(old_tail, new_tail,
                                          std::memory_order_acq_rel,
                                          std::memory_order_relaxed));

    pool_[GetIndex(old_tail)] = element;
    do
    {
        old_commit = old_tail;
    } while (!commit_.compare_exchange_weak(old_commit, new_tail,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed));

    wait_strategy_->NotifyOne();
    return true;
}

template <typename T>
bool BoundedQueue<T>::Enqueue(T &&element)
{
    uint64_t new_tail = 0;
    uint64_t old_commit = 0;
    uint64_t old_tail = tail_.load(std::memory_order_acquire);

    do
    {
        new_tail = old_tail + 1;
        if (GetIndex(new_tail) == GetIndex(head_.load(std::memory_order_acquire)))
        {
            return false;
        }
    } while (!tail_.compare_exchange_weak(old_tail, new_tail,
                                          std::memory_order_acq_rel,
                                          std::memory_order_relaxed));

    pool_[GetIndex(old_tail)] = std::move(element);
    do
    {
        old_commit = old_tail;
    } while (!commit_.compare_exchange_weak(old_commit, new_tail,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed));
    wait_strategy_->NotifyOne();
    return true;
}

template <typename T>
bool BoundedQueue<T>::Dequeue(T *element)
{
    uint64_t new_head = 0;
    uint64_t old_head = head_.load(std::memory_order_acquire);
    do
    {
        new_head = old_head + 1;
        if (new_head == commit_.load(std::memory_order_acquire))
        {
            return false;
        }
        *element = pool_[GetIndex(new_head)];
    } while (!head_.compare_exchange_weak(old_head, new_head,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire));
    return true;
}

template <typename T>
bool BoundedQueue<T>::WaitEnqueue(const T &element)
{
    while (!notify_all_)
    {
        if (Enqueue(element))
        {
            return true;
        }
        if (wait_strategy_->EmptyWait())
        {
            continue;
        }
        break;
    }
    return false;
}

template <typename T>
bool BoundedQueue<T>::WaitEnqueue(T &&element)
{
    while (!notify_all_)
    {
        if (Enqueue(std::move(element)))
        {
            return true;
        }
        if (wait_strategy_->EmptyWait())
        {
            continue;
        }
        break;
    }
    return false;
}

template <typename T>
inline bool BoundedQueue<T>::WaitDequeue(T *element)
{
    while (!notify_all_)
    {
        if (Dequeue(element))
        {
            return true;
        }
        if (wait_strategy_->EmptyWait())
        {
            continue;
        }
        break;
    }
    return false;
}

template <typename T>
uint64_t BoundedQueue<T>::Size()
{
    return tail_ - head_ - 1;
}

template <typename T>
bool BoundedQueue<T>::Empty()
{
    return Size() == 0;
}

template <typename T>
uint64_t BoundedQueue<T>::Head()
{
    return head_.load();
}

template <typename T>
uint64_t BoundedQueue<T>::Tail()
{
    return tail_.load();
}

template <typename T>
uint64_t BoundedQueue<T>::Commit()
{
    return commit_.load();
}

template <typename T>
void BoundedQueue<T>::SetWaitStrategy(WaitStrategy *wait_strategy)
{
    wait_strategy_.reset(wait_strategy);
}

template <typename T>
inline void BoundedQueue<T>::BreakAllWait()
{
    notify_all_ = true;
    wait_strategy_->NotifyAll();
}

template <typename T>
uint64_t BoundedQueue<T>::GetIndex(uint64_t num)
{
    return num - (num / pool_size_) * pool_size_;
}

#endif

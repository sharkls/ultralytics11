#ifndef QUEUE_DEQUE_QUEUE_HPP
#define QUEUE_DEQUE_QUEUE_HPP

#include <mutex>
#include <condition_variable>
#include <deque>

#include "include/common/log.hpp"

template <typename T>
class SafeDataQueue
{
public:
    using BufferType = std::shared_ptr<T>;

    SafeDataQueue(const uint32_t size);
    ~SafeDataQueue();

    bool Full();
    bool Empty();

    void FillFront(const BufferType& value);
    bool FetchBack(BufferType& value, uint32_t timeout = -1);
    
    void FillBack(const BufferType& value);
    bool FetchFront(BufferType& value, uint32_t timeout = -1);
    

private:
    SafeDataQueue(const SafeDataQueue&) = delete;
    SafeDataQueue& operator=(const SafeDataQueue&) = delete;

    bool Wait(std::unique_lock<std::mutex>& lock, uint32_t timeout = -1);

private:
    std::deque<BufferType> buffer_;
    std::condition_variable cv_;
    std::mutex mutex_;
    int capacity_;
};

template <typename T>
SafeDataQueue<T>::SafeDataQueue(const uint32_t size)
{
    capacity_ = size;
}

template <typename T>
SafeDataQueue<T>::~SafeDataQueue()
{
}

template <typename T>
inline bool SafeDataQueue<T>::Full()
{
    std::unique_lock<std::mutex> lock(mutex_);
    return buffer_.size() == capacity_;
}

template <typename T>
inline bool SafeDataQueue<T>::Empty()
{
    std::unique_lock<std::mutex> lock(mutex_);
    return buffer_.empty();
}

template <typename T>
inline void SafeDataQueue<T>::FillFront(const BufferType &value)
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (buffer_.size() >= capacity_)
    {
        buffer_.pop_front();
    }
    buffer_.push_front(value);
    lock.unlock();
    cv_.notify_one();
}

template <typename T>
bool SafeDataQueue<T>::FetchBack(BufferType &value, uint32_t timeout)
{   
    std::unique_lock<std::mutex> lock(mutex_);
    if(!Wait(lock, timeout))
    {
        return false;
    }
    value = buffer_.back();
    buffer_.pop_back();
    return true;
}

template <typename T>
void SafeDataQueue<T>::FillBack(const BufferType &value)
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (buffer_.size() >= capacity_)
    {
        buffer_.pop_front();
    }
    buffer_.push_back(value);
    lock.unlock();
    cv_.notify_one();
}

template <typename T>
bool SafeDataQueue<T>::FetchFront(BufferType &value, uint32_t timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if(!Wait(lock, timeout))
    {
        return false;
    }
    value = buffer_.front();
    buffer_.pop_front();
    return true;
}

template <typename T>
inline bool SafeDataQueue<T>::Wait(std::unique_lock<std::mutex>& lock, uint32_t timeout)
{
    if (timeout < 0)
    {
        cv_.wait(lock, [&](){return !buffer_.empty();});
    }
    else if (timeout == 0)
    {
        if (buffer_.empty())
        {
            return false;
        }
    }
    else
    {
        bool ok = cv_.wait_for(lock, std::chrono::milliseconds(timeout), [&](){return !buffer_.empty();});
        if (!ok)
        {
            return false;
        }
    }
    return true;
}

#endif

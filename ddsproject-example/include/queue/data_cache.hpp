#ifndef CACHE_DATA_CACHE_HPP
#define CACHE_DATA_CACHE_HPP

#include <vector>
#include <mutex>

#include "include/common/log.hpp"

template<typename T>
class CacheBuffer
{
public:
    explicit CacheBuffer(uint64_t size);

    T& operator[](const uint64_t& pos);
    const T& at(const uint64_t& pos);

    uint64_t Head();
    uint64_t Tail();
    uint64_t Size();

    const T& Front();
    const T& Back();

    bool Full() const;
    bool Empty() const;

    void Fill(const T& value);

    std::mutex& Mutex();

private:
    CacheBuffer& operator=(const CacheBuffer&) = delete;
    CacheBuffer(const CacheBuffer&) = delete;

    uint64_t GetIndex(const uint64_t& pos) {
        return pos % capacity_;
    }

private:
    uint32_t capacity_;
    std::vector<T> buffer_;

    uint64_t head_{0};
    uint64_t tail_{0};
    
    mutable std::mutex mutex_;
};

template <typename T>
CacheBuffer<T>::CacheBuffer(uint64_t size)
{
    capacity_ = size + 1;
    buffer_.resize(capacity_);
}

template <typename T>
T &CacheBuffer<T>::operator[](const uint64_t &pos)
{
    return buffer_[GetIndex(pos)];
}

template <typename T>
const T &CacheBuffer<T>::at(const uint64_t &pos)
{
    return buffer_[GetIndex(pos)];
}

template <typename T>
uint64_t CacheBuffer<T>::Head()
{
    return head_ + 1;
}

template <typename T>
uint64_t CacheBuffer<T>::Tail()
{
    return tail_;
}

template <typename T>
inline uint64_t CacheBuffer<T>::Size()
{
    return tail_ - head_;
}

template <typename T>
const T &CacheBuffer<T>::Front()
{
    return buffer_[GetIndex(head_ + 1)] ;   
}

template <typename T>
const T &CacheBuffer<T>::Back()
{
    return buffer_[GetIndex(tail_)];
}

template <typename T>
bool CacheBuffer<T>::Full() const
{
    return capacity_ - 1 == tail_ - head_;
}

template <typename T>
bool CacheBuffer<T>::Empty() const
{
    return tail_ == 0;
}

template <typename T>
void CacheBuffer<T>::Fill(const T &value)
{
    if (Full())
    {
        buffer_[GetIndex(head_)] = value; 
        ++head_;
        ++tail_;
    }
    else
    {
        buffer_[GetIndex(tail_ + 1)] = value;
        ++tail_;
    }
}

template <typename T>
std::mutex &CacheBuffer<T>::Mutex()
{
    return mutex_;    
}

#endif
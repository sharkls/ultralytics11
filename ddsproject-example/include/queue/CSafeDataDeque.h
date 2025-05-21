/*******************************************************
 文件名：CSafeDataDeque.h
 作者：
 描述：线程安全的数据体队列
 版本：v1.0
 日期：2020-03-04
 *******************************************************/

#ifndef CSAFEDATADEQUE_H
#define CSAFEDATADEQUE_H

#include <mutex>
#include <thread>
#include <vector>
#include <map>
#include <deque>
#include <iostream>
#include <condition_variable>
#include "include/common/log.hpp"

struct LicenseData;

// template<typename T, bool (*CompareFun)(const T& p_constRef)>;

//线程安全队列，默认最大长度10
template<typename T, typename Container = std::deque<T>>
class __attribute__ ((visibility("default"))) CSafeDataDeque
{
public:
	CSafeDataDeque()
	{
		m_pCommpareFun = NULL;
		m_pReleaseFun = NULL;
		m_unMaxSize = 2;
	}

	virtual ~CSafeDataDeque()
	{
		//捕获异常，并提示
		try
		{
			Release();
		}
		catch(const std::exception& e)
		{
			std::cerr << "Exception--CSafeDataDeque::" << e.what() << '\n';
		}
		catch(const std::string& e)
		{
			std::cerr << "Exception--CSafeDataDeque::" << e << '\n';
		}
	}

	// T& operator[](std::size_t l_nIndex)
	// {
	// 	std::unique_lock<std::mutex> lock(m_Mutex);
	// 	return m_dqData[l_nIndex];
	// }

	//是否为空
	bool Empty()
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		bool bEmpty = m_dqData.empty();
		return bEmpty;
	}

	// template <typename Element>
	// void Push(Element&& element) {
	// 	std::lock_guard<std::mutex> lock(mutex_);
	// 	m_dqData.push(std::forward<Element>(element));
	// 	m_Cv.notify_one();
	// }

	//在队列尾部追加，超过队列最大长度，则从头部弹出元素，直到<=最大长度，然后才唤醒其他线程
	void PushBack(const T& p_refData)
	{
		// LOG(INFO) << "PushBack(const T& p_refData) step 1" << std::endl;
		// auto l_start = std::chrono::high_resolution_clock::now();
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_dqData.push_back(p_refData);
		while (m_dqData.size() > m_unMaxSize)
		{
			if(m_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素
			{
				T l_data = m_dqData.front();
				m_pReleaseFun(l_data);
			}

			//从队列中移除
			m_dqData.pop_front();
		}
		lock.unlock();
		m_Cv.notify_one();
		// auto l_end = std::chrono::high_resolution_clock::now();
		// auto l_nMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(l_end-l_start).count();
		// LOG(INFO) << "PushBack(const T& p_refData) step 1, l_nMicroseconds=" << l_nMicroseconds << std::endl;
	}

	void PushBack(T&& p_refData)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_dqData.push_back(std::move(p_refData));
		while (m_dqData.size() > m_unMaxSize)
		{
			if(m_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素
			{
				T l_data = m_dqData.front();
				m_pReleaseFun(l_data);
			}

			//从队列中移除
			m_dqData.pop_front();
		}
		lock.unlock();
		m_Cv.notify_one();
	}

	//在队列头部添加，超过队列最大长度，则从头部弹出元素，直到<=最大长度，然后才唤醒其他线程
	void PushFront(const T& p_refData)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_dqData.push_front(p_refData);
		while (m_dqData.size() > m_unMaxSize)
		{
			if(m_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素
			{
				T l_data = m_dqData.front();
				m_pReleaseFun(l_data);
			}

			//从队列中移除
			m_dqData.pop_front();
		}
		lock.unlock();
		m_Cv.notify_one();
	}

	void PushFront(T&& p_refData)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_dqData.push_front(std::move(p_refData));
		while (m_dqData.size() > m_unMaxSize)
		{
			if(m_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素
			{
				T l_data = m_dqData.front();
				m_pReleaseFun(l_data);
			}

			//从队列中移除
			m_dqData.pop_front();
		}
		lock.unlock();
		m_Cv.notify_one();
	}

	//获取队列尾部元素的一份拷贝，并从队列弹出该元素, p_nTimeOutMs单位ms，默认-1，表示阻塞模式，=0表示非阻塞模式，>0表示超时模式，超时后若队列无数据，会返回false，否则返回true
	bool PopBack(T& p_refData, const int32_t& p_nTimeOut=-1)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(p_nTimeOut<0)
		{
			m_Cv.wait(lock, [&]() {return !m_dqData.empty();});
		}
		else if(0 == p_nTimeOut)
		{
			if(m_dqData.empty())
				return false;
		}
		else
		{
			bool l_bOk = m_Cv.wait_for(lock, std::chrono::milliseconds(p_nTimeOut), [&]() {return !m_dqData.empty();});
			if(!l_bOk)
			{
				return false;
			}
		}

		p_refData = std::move(m_dqData.back());
		m_dqData.pop_back();
		return true;
	}

	//获取队列头部元素的一份拷贝，并从队列弹出该元素, p_nTimeOutMs单位ms，默认-1，表示阻塞模式，=0表示非阻塞模式，>0表示超时模式，超时后若队列无数据，会返回false，否则返回true
	bool PopFront(T& p_refData, const int32_t& p_nTimeOut=-1)
	{
		// LOG(INFO) << "PopFront step 1" << std::endl;
		// auto l_start = std::chrono::high_resolution_clock::now();
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(p_nTimeOut<0)
		{
			m_Cv.wait(lock, [&]() {return !m_dqData.empty();});
		}
		else if(0 == p_nTimeOut)
		{
			if(m_dqData.empty())
				return false;
		}
		else
		{
			bool l_bOk = m_Cv.wait_for(lock, std::chrono::milliseconds(p_nTimeOut), [&]() {return !m_dqData.empty();});
			if(!l_bOk)
			{
				return false;
			}
		}
		p_refData = std::move(m_dqData.front());
		m_dqData.pop_front();
		// auto l_end = std::chrono::high_resolution_clock::now();
		// auto l_nMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(l_end-l_start).count();
		// LOG(INFO) << "PopFront step 3, l_nMicroseconds=" << l_nMicroseconds << std::endl;
		return true;
	}

	//清空队列，只清空队列空间，（若队列元素为指针，不做释放指针元素空间的操作）
	void Clear()
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_dqData.clear();
	}

	//获取尾部元素的引用, p_nTimeOutMs单位ms，默认-1，表示阻塞模式，=0表示非阻塞模式，>0表示超时模式，超时后若队列无数据，会抛出异常
	T& Back(const int32_t& p_nTimeOut=-1) 
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(p_nTimeOut<0)
			m_Cv.wait(lock, [&]() {return !m_dqData.empty();});
		else if(0 == p_nTimeOut)
		{
			if(m_dqData.empty())
				throw std::string("ERROR: The deque is empty!");
		}
		else
		{
			bool l_bOk = m_Cv.wait_for(lock, std::chrono::milliseconds(p_nTimeOut), [&]() {return !m_dqData.empty();});
			if(!l_bOk)
			{
				throw std::string("ERROR: The deque is empty!");
			}
		}
		return m_dqData.back();
	}

	//获取头部元素的引用, p_nTimeOutMs单位ms，默认-1，表示阻塞模式，=0表示非阻塞模式，>0表示超时模式，超时后若队列无数据，会抛出异常
	T& Front(const int32_t& p_nTimeOut=-1) 
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(p_nTimeOut<0)
			m_Cv.wait(lock, [&]() {return !m_dqData.empty();});
		else if(0 == p_nTimeOut)
		{
			if(m_dqData.empty())
				throw std::string("ERROR: The deque is empty!");
		}
		else
		{
			bool l_bOk = m_Cv.wait_for(lock, std::chrono::milliseconds(p_nTimeOut), [&]() {return !m_dqData.empty();});
			if(!l_bOk)
			{
				throw std::string("ERROR: The deque is empty!");
			}
		}
		return m_dqData.front();
	}

	//队列的长度
	std::size_t Size()
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		std::size_t l_nSize = m_dqData.size();
		return l_nSize;
	}

	//定义释放元素空间的函数原型
    typedef void (*ReleaseFun)(T& p_constRef);

	//定义比较元素是否相同的函数原型
    typedef bool (*CompareFun)(const T& p_constRef1, const T& p_constRef2);

	//定义判断元素是否满足相关条件的函数原型
    typedef bool (*Condition)(T const & p_constRef);

    typedef bool (*ConditionData)(T const & p_constRef,LicenseData* p_obj);

	//从队列中移除指定元素
	//如果传入了元素比较函数的指针，则使用该函数比较元素，否则按默认==比较
	//如果传入了释放函数的指针，则使用该函数释放元素，否则只是从队列中移除
	bool EraseByValue(T& p_refData, CompareFun p_CompareFun=NULL, ReleaseFun p_pReleaseFun=NULL)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(m_dqData.empty())
			return false;

		if(p_CompareFun)//如果传入了比较函数的指针，则使用该函数比较元素
		{
			if(p_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素，并从队列中移除
			{
				for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
				{
					if(p_CompareFun(p_refData, *l_it))
					{
						p_pReleaseFun(*l_it);
						l_it = m_dqData.erase(l_it);
					}
					else
					{
						l_it++;
					}
				}
			}
			else//否则只从队列中移除
			{
				for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
				{
					if(p_CompareFun(p_refData, *l_it))
					{
						l_it = m_dqData.erase(l_it);
					}
					else
					{
						l_it++;
					}
				}	
			}
		}
		else//否则按默认==比较
		{
			if(p_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素，并从队列中移除
			{
				for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
				{
					if(p_refData == *l_it)
					{
						p_pReleaseFun(*l_it);
						l_it = m_dqData.erase(l_it);
					}
					else
					{
						l_it++;
					}
				}
			}
			else//否则只从队列中移除
			{
				for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
				{
					if(p_refData == *l_it)
					{
						l_it = m_dqData.erase(l_it);
					}
					else
					{
						l_it++;
					}
				}	
			}
		}
		return true;
	}

	//从队列中移除所有满足指定条件的元素
	//必须传入条件函数的指针，使用该函数元素是否满足该条件
	//如果传入了释放函数的指针，则使用该函数释放元素，否则只是从队列中移除
	bool EraseByCondition(Condition p_pConditionFun, ReleaseFun p_pReleaseFun=NULL)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(m_dqData.empty() || !p_pConditionFun)
			return false;

		if (!p_pReleaseFun)
			p_pReleaseFun = m_pReleaseFun;
		
		if(p_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素，并从队列中移除
		{
			for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
			{
				if(p_pConditionFun(*l_it))
				{
					p_pReleaseFun(*l_it);
					l_it = m_dqData.erase(l_it);
				}
				else
				{
					l_it++;
				}
			}
		}
		else//否则只从队列中移除
		{
			for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
			{
				if(p_pConditionFun(*l_it))
				{
					l_it = m_dqData.erase(l_it);
				}
				else
				{
					l_it++;
				}
			}	
		}
	
		return true;
	}

	//从队列中获取所有满足条件的元素
	//传入条件函数的指针，使用该函数判断元素是否满足该条件
	//若传入函数指针为NULL，则获取拷贝整个的所有元素
	void GetByCondition(std::deque<T>& p_dqElement, Condition p_pConditionFun=NULL)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(m_dqData.empty())
			return;

		if(p_pConditionFun)
		{
			for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
			{
				if(p_pConditionFun(*l_it))
				{
					p_dqElement.push_back(*l_it);
					l_it = m_dqData.erase(l_it);
				}
				else
				{
					l_it++;
				}
			}
		}
		else
		{
			p_dqElement.assign(m_dqData.begin(), m_dqData.end());
			m_dqData.clear();
		}
	}

	void GetByConditionData(T& element, ConditionData p_pConditionFun,LicenseData* pobj)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(m_dqData.empty())
			return;


		if(p_pConditionFun)
		{
			for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
			{
				if(p_pConditionFun(*l_it,pobj))
				{
					element = *l_it;
					l_it = m_dqData.erase(l_it);
				}
				else
				{
					l_it++;
				}
			}
		}
		else
		{
			m_dqData.clear();
		}
	}

	//清空并释放本队列及所有元素空间
	void Release() //指定异常参数表，则只能抛出指定类型的异常
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		if(m_pReleaseFun)//如果传入了释放函数的指针，则使用该函数释放元素，并从队列中移除
		{
			for (auto l_it = m_dqData.begin(); l_it != m_dqData.end();)
			{
				m_pReleaseFun(*l_it);//使用时需注意，如果队列元素是指针，且有两个位置存放的是同一个指针，则此处可能会有重复释放的问题
				l_it = m_dqData.erase(l_it);
			}
		}
		else
		{
			m_dqData.clear();

			//抛出异常
			// throw std::string("The resource release function is not specified, please confirm whether the resource has been released");

			//输出提示
			LOG(WARNING) << "The resource release function is not specified, please confirm whether the resource has been released!" << std::endl;
		}
	}

	//设置比较元素是否相同的函数指针
	void SetCommpareFun(CompareFun p_pCommpareFun)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_pCommpareFun = p_pCommpareFun;
	}

	//设置释放元素空间的函数指针
	void SetReleaseFun(ReleaseFun p_pReleaseFun)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_pReleaseFun = p_pReleaseFun;
	}

	//队列最大长度，超过该长度，则从头部弹出元素，直到<=该长度
	void SetMaxSize(const uint32_t& p_unMaxSize)
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_unMaxSize = p_unMaxSize;
	}

private:
	CSafeDataDeque(const CSafeDataDeque& p_refOther){}
	CSafeDataDeque& operator=(const CSafeDataDeque& p_refDq){}

protected:
	Container				m_dqData;			//数据体队列
	mutable std::mutex		m_Mutex;			//队列锁
	std::condition_variable m_Cv;				//条件变量
	CompareFun				m_pCommpareFun;		//比较元素是否相同的函数指针
	ReleaseFun				m_pReleaseFun;		//释放元素空间的函数
	uint32_t				m_unMaxSize;		//队列最大长度，超过该长度，则从头部弹出元素，知道<=m_unMaxSize
};

#endif
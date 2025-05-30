#ifndef COMMON_MACROS_HPP
#define COMMON_MACROS_HPP

#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#define DEFINE_TYPE_TRAIT(name, func)                        \
    template <typename T>                                    \
    struct name                                              \
    {                                                        \
        template <typename Class>                            \
        static constexpr bool Test(decltype(&Class::func) *) \
        {                                                    \
            return true;                                     \
        }                                                    \
                                                             \
        template <typename>                                  \
        static constexpr bool Test(...)                      \
        {                                                    \
            return false;                                    \
        }                                                    \
        static constexpr bool value = Test<T>(nullptr);      \
    };                                                       \
    template <typename T>                                    \
    constexpr bool name<T>::value;

DEFINE_TYPE_TRAIT(HasShutdown, Shutdown)

template <typename T>
typename std::enable_if<HasShutdown<T>::value>::type CallShutdown(T *instance)
{
    instance->Shutdown();
}

template <typename T>
typename std::enable_if<!HasShutdown<T>::value>::type CallShutdown(T *instance)
{
    (void *)instance;
}

#undef UNUSED
#undef DISALLOW_COPY_AND_ASSIGN

#define UNUSED(param) (void)param

// 禁止拷贝构造和复制构造
#define DISALLOW_COPY_AND_ASSIGN(classname) \
    classname(const classname &) = delete;  \
    classname &operator=(const classname &) = delete;

#define DECLARE_SINGLETON(classname)                                                  \
public:                                                                               \
    static classname *Instance(bool create_if_needed = true)                          \
    {                                                                                 \
        static classname *instance = nullptr;                                         \
        if (!instance && create_if_needed)                                            \
        {                                                                             \
            static std::once_flag flag;                                               \
            std::call_once(flag, [&] { instance = new (std::nothrow) classname(); }); \
        }                                                                             \
        return instance;                                                              \
    }                                                                                 \
                                                                                      \
    static void CleanUp()                                                             \
    {                                                                                 \
        auto instance = Instance(false);                                              \
        if (instance != nullptr)                                                      \
        {                                                                             \
            CallShutdown(instance);                                                   \
        }                                                                             \
    }                                                                                 \
                                                                                      \
private:                                                                              \
    classname();                                                                      \
    DISALLOW_COPY_AND_ASSIGN(classname)


// cpu pause,汇编语言，将当前cpu休眠，用于日志系统的写入与刷新
inline static void cpu_relax()
{
#if defined(__aarch64__)
    asm volatile("rep; ");
#else
    asm volatile("rep; nop" ::: "memory");
#endif
}

#endif


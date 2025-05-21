#ifndef CLASS_LOADER_SHARED_LIBRARY_SHARED_LIBRARY_HPP
#define CLASS_LOADER_SHARED_LIBRARY_SHARED_LIBRARY_HPP

#include <string>
#include <mutex>
#include <dlfcn.h>

#include "include/class_loader/shared_library/exception.hpp"

class SharedLibrary
{
public:
    enum Flags
    {
        SHLIB_GLOBAL = 1,
        SHLIB_LOCAL = 2,
    };

    SharedLibrary() = default;
    ~SharedLibrary();

    explicit SharedLibrary(const std::string& path);
    SharedLibrary(const std::string& path, int flags);

    void Load(const std::string& path);
    void Load(const std::string& path, int flags);

    void Unload();

    bool IsLoaded();

    bool HasSymbol(const std::string& name);

    void* GetSymbol(const std::string& name);

    const std::string& GetPath() const;

    SharedLibrary(const SharedLibrary&) = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

private:
    void* handle_ = nullptr;
    std::string path_;
    std::mutex mutex_;
};


#endif
#ifndef COMMON_ENVIRONMENT_HPP
#define COMMON_ENVIRONMENT_HPP

#include <string>

#include "include/common/log.hpp"

inline std::string GetEnv(const std::string &var_name,
                          const std::string &default_value= "")
{
    const char *var = std::getenv(var_name.c_str());
    if (var == nullptr)
    {
        TWARN << "Environment variable [" << var_name << "] not set, fallback to" << default_value;
        return default_value;
    }
    return std::string(var);
}

inline const std::string WorkRoot()
{
    std::string work_root = GetEnv("DDS_PATH");
    if (work_root.empty()) {
        work_root = "/workspace/ddsproject";
    }
    return work_root;
}

#endif
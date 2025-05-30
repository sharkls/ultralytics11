#ifndef MAINBOARD_MODULE_ARGUMENT_HPP
#define MAINBOARD_MODULE_ARGUMENT_HPP

#include <list>
#include <libgen.h>
#include <getopt.h>

#include "include/common/log.hpp"

class ModuleArgument
{
public:
    ModuleArgument() = default;
    ~ModuleArgument() = default;

    void DisplayUsage();
    void ParseArgument(int argc, char* const argv[]);
    void GetOptions(const int argc, char* const argv[]);
    const std::list<std::string>& GetDagConfList() const;

private:
    std::string binary_name_;
    std::list<std::string> dag_conf_list_;
};

#endif
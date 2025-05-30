#ifndef COMMANDLINE_ACTIVITY_ACTIVITY_ARGUMENT_HPP
#define COMMANDLINE_ACTIVITY_ACTIVITY_ARGUMENT_HPP

#include <getopt.h>

#include "include/common/log.hpp"
#include "include/common/global_data.hpp"

class ActivityArgument
{
public:
    ActivityArgument() = default;
    ~ActivityArgument() = default;

    void DisplayUsage();
    void ParseArgument(int argc, char* const argv[]);
    const std::string& GetConfigFile() const;

private:
    std::string conf_file_;
};

#endif
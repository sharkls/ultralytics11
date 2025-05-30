#ifndef COMMANDLINE_MASTER_MASTER_ARGUMENT_HPP
#define COMMANDLINE_MASTER_MASTER_ARGUMENT_HPP

#include <getopt.h>

#include "include/common/log.hpp"
#include "include/common/global_data.hpp"

class MasterArgument
{
public:
    MasterArgument() = default;
    ~MasterArgument() = default;

    void DisplayUsage();
    void ParseArgument(int argc, char* const argv[]);
};

#endif
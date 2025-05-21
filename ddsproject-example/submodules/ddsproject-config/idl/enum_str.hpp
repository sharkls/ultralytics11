#ifndef IDL_ENUM_STR_HPP
#define IDL_ENUM_STR_HPP

#include "idl/master_activity_message/master_activity_message.h"

#include <string>

static std::string GetActivityRunCmdStr(const ActivityRunCmdEnum &cmd)
{
    std::string cmd_str;
    switch (cmd)
    {
    case RUN:
        cmd_str = "run";
        break;
    case PAUSE:
        cmd_str = "pause";
        break;
    case SHUTDOWN:
        cmd_str = "shutdown";
        break;
    }
    return cmd_str;
}

static std::string GetActivityStatusStr(const ActivityStatusEnum &status)
{
    std::string status_str;
    switch (status)
    {
    case INIT:
        status_str = "init";
        break;
    case RUNNING:
        status_str = "running";
        break;
    case PAUSED:
        status_str = "pause";
        break;
    case FINISHED:
        status_str = "finished";
        break;
    }
    return status_str;
}

#endif
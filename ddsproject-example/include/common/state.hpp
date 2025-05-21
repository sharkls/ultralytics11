#ifndef COMMON_STATE_HPP
#define COMMON_STATE_HPP

#include <atomic>
#include <thread>

enum class State
{
    STATE_UNINITIALIZED,
    STATE_INITIALIZED,
    STATE_RUNNING,
    STATE_SHUTDOWN,
};

State GetState();
void SetState(const State& state);

bool OK();

bool IsShutdown();

void WaitForShutdown();

#endif
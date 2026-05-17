#pragma once

#include "tracing.h"
#include <atomic>

struct WorkerContext {
    _Thread thread;
    std::atomic<bool> running;
};

void worker_start(WorkerContext* ctx);
void worker_stop(WorkerContext* ctx);

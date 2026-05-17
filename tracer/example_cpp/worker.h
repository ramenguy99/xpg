#pragma once

#include <atomic>
#include <thread>

struct WorkerContext {
    std::thread thread;
    std::atomic<bool> running;
};

void worker_start(WorkerContext* ctx);
void worker_stop(WorkerContext* ctx);

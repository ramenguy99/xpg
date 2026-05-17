#include <cstdio>
#include <chrono>
#include <thread>

#include "worker.h"

#include "tracer.h"

TRACEPOINT_DEFINE(tp_worker, "worker.tick");

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

static void worker_proc(WorkerContext* ctx) {
    int tick = 0;

    printf("[worker] started\n");

    while (ctx->running.load(std::memory_order_relaxed)) {
        float value = 0.5f + 0.5f * (tick % 10);

        TRACE(tp_worker,
            TI64("tick", tick),
            TF64("value", value),
            TI64("healthy", true)
        );

        tick++;

        sleep_ms(500);
    }

    printf("[worker] stopped after %d ticks\n", tick);
}

void worker_start(WorkerContext* ctx) {
    ctx->running.store(true, std::memory_order_relaxed);
    ctx->thread = std::thread(worker_proc, ctx);
}

void worker_stop(WorkerContext* ctx) {
    ctx->running.store(false, std::memory_order_relaxed);
    if (ctx->thread.joinable())
        ctx->thread.join();
}

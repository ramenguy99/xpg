#include <cstdio>
#include <ctime>
#include <cerrno>

#include "worker.h"

TRACEPOINT_DEFINE(tp_worker, "worker.tick");

static THREAD_PROC(worker_proc) {
    WorkerContext* ctx = (WorkerContext*)data;
    int tick = 0;

    printf("[worker] started\n");

    while (ctx->running.load(std::memory_order_relaxed)) {
        float value = 0.5f + 0.5f * (tick % 10);

        TRACE(tp_worker,
            TI32("tick", tick),
            TF32("value", value),
            TBOOL("healthy", true)
        );

        tick++;

        struct timespec ts = { 0, 500000000L };
        while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
    }

    printf("[worker] stopped after %d ticks\n", tick);
    return 0;
}

void worker_start(WorkerContext* ctx) {
    ctx->running.store(true, std::memory_order_relaxed);
    create_thread(worker_proc, ctx, &ctx->thread);
}

void worker_stop(WorkerContext* ctx) {
    ctx->running.store(false, std::memory_order_relaxed);
    join_thread(&ctx->thread);
}

#include <stdio.h>

#include "worker.h"

TRACEPOINT_DEFINE(tp_worker, "worker.tick");

static THREAD_PROC(worker_proc) {
    WorkerContext* ctx = (WorkerContext*)data;
    int tick = 0;

    printf("[worker] started\n");

    while (atomic_load_explicit(&ctx->running, memory_order_relaxed)) {
        float value = 0.5f + 0.5f * (tick % 10);

        TRACE(tp_worker,
            TI64("tick", tick),
            TF64("value", value),
            TI64("healthy", 1)
        );

        tick++;
        thread_sleep_ms(500);
    }

    printf("[worker] stopped after %d ticks\n", tick);
    return 0;
}

void worker_start(WorkerContext* ctx) {
    atomic_store_explicit(&ctx->running, 1, memory_order_relaxed);
    create_thread(worker_proc, ctx, &ctx->thread);
}

void worker_stop(WorkerContext* ctx) {
    atomic_store_explicit(&ctx->running, 0, memory_order_relaxed);
    join_thread(&ctx->thread);
}

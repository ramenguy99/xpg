#include <cstdio>
#include <ctime>
#include <cerrno>

#include "worker.h"

TRACEPOINT_DEFINE(tp_worker, "worker.tick");

static void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
#endif
}

static _THREAD_PROC(worker_proc) {
    WorkerContext* ctx = (WorkerContext*)data;
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
    return 0;
}

void worker_start(WorkerContext* ctx) {
    ctx->running.store(true, std::memory_order_relaxed);
    _create_thread(worker_proc, ctx, &ctx->thread);
}

void worker_stop(WorkerContext* ctx) {
    ctx->running.store(false, std::memory_order_relaxed);
    _join_thread(&ctx->thread);
}

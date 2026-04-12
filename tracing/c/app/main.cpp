#include <cstdio>
#include <csignal>
#include <ctime>
#include <cerrno>

#define TRACING_IMPLEMENTATION
#include "tracing.h"
#include "worker.h"

TRACEPOINT_DEFINE(tp_main, "main.lifecycle");

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int) {
    g_running = 0;
}

static void sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

int main() {
    tracer_init();

    int tcp = tracer_add_tcp_subscriber("127.0.0.1", 9168);
    tracer_subscribe_all(tcp);

    printf("Tracing started (tcp=%d)\n", tcp);

    TRACE(tp_main,
        TI32("status", 1),
        TSTR("event", "started", 7)
    );

    // Start worker thread
    WorkerContext worker = {};
    worker_start(&worker);

    // Wait for Ctrl+C
    signal(SIGINT, signal_handler);
    printf("Running... press Ctrl+C to stop\n");

    while (g_running) {
        sleep_ms(100);
    }

    printf("\nShutting down...\n");

    TRACE(tp_main,
        TI32("status", 0),
        TSTR("event", "stopping", 8)
    );

    // Stop worker, then close tracing
    worker_stop(&worker);
    tracer_close();

    printf("Done.\n");
    return 0;
}

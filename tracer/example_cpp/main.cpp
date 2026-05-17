#include <cstdio>
#include <csignal>
#include <ctime>
#include <cerrno>
#include <thread>

#include "tracer.h"
#include "worker.h"

TRACEPOINT_DEFINE(tp_main, "main.lifecycle");

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int) {
    g_running = 0;
}

int main() {
    tracer_init();

    int tcp = tracer_add_tcp_subscriber("127.0.0.1", 9168);
    tracer_subscribe_all(tcp);

    SqliteConfig cfg = sqlite_config_default();
    int db = tracer_add_sqlite_subscriber("trace.db", &cfg);
    tracer_subscribe_all(db);

    printf("Tracing started (tcp=%d, sqlite=%d)\n", tcp, db);

    TRACE(tp_main,
        TI64("status", 1),
        TSTR("event", "started", 7)
    );

    // Start worker thread
    WorkerContext worker = {};
    worker_start(&worker);

    // Wait for Ctrl+C
    signal(SIGINT, signal_handler);
    printf("Running... press Ctrl+C to stop\n");

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    printf("\nShutting down...\n");

    TRACE(tp_main,
        TI64("status", 0),
        TSTR("event", "stopping", 8)
    );

    // Stop worker, then close tracing
    worker_stop(&worker);
    tracer_close();

    printf("Done.\n");
    return 0;
}

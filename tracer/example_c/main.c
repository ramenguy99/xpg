#include <stdio.h>
#include <signal.h>

#include "worker.h"

TRACEPOINT_DEFINE(tp_main, "main.lifecycle");

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

int main(void) {
    tracer_init();

    int tcp = tracer_add_tcp_subscriber("127.0.0.1", 9168);
    tracer_subscribe_all(tcp);

    printf("Tracing started (tcp=%d)\n", tcp);

    TRACE(tp_main,
        TI64("status", 1),
        TSTR("event", "started", 7)
    );

    WorkerContext worker = {0};
    worker_start(&worker);

    signal(SIGINT, signal_handler);
    printf("Running... press Ctrl+C to stop\n");

    while (g_running) {
        thread_sleep_ms(100);
    }

    printf("\nShutting down...\n");

    TRACE(tp_main,
        TI64("status", 0),
        TSTR("event", "stopping", 8)
    );

    worker_stop(&worker);
    tracer_close();

    printf("Done.\n");
    return 0;
}

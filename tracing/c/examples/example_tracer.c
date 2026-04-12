#include <stdlib.h>

#ifdef TRACER_SQLITE_ENABLED
#include <sqlite3.h>
#endif

#define TRACING_IMPLEMENTATION
#include "tracing.h"

TRACEPOINT_DEFINE(tp_physics, "physics.step");
TRACEPOINT_DEFINE(tp_render,  "render.frame");

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

int main(void) {
    tracer_init();

    int tcp = tracer_add_tcp_subscriber("127.0.0.1", 9168);
    tracer_subscribe_all(tcp);

#ifdef TRACER_SQLITE_ENABLED
    int sql = tracer_add_sqlite_subscriber("traces.db");
    tracer_subscribe_all(sql);
    printf("SQLite subscriber added\n");
#endif

    printf("Tracing started (tcp=%d). Sending 100 events...\n", tcp);

    for (int i = 0; i < 100; i++) {
        float dt = 0.016f;

        TRACE(tp_physics,
            TF32("dt", dt),
            TI32("step", i)
        );

        float image_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        size_t shape[] = { 2, 2 };
        size_t strides[] = { 2 * sizeof(float), sizeof(float) };

        TRACE(tp_render,
            TU64("width", 2),
            TU64("height", 2),
            TNDARRAY("image", 2, shape, strides, image_data, sizeof(float), "<f4")
        );

        sleep_ms(10);
    }

    printf("Done tracing, closing...\n");
    tracer_close();
    printf("Closed.\n");

    return 0;
}

#include <stdlib.h>

#define TRACING_IMPLEMENTATION
#include "tracing.h"

void fatal(const char* msg, Result result) {
    printf("%s: %d\n", msg, result);
    exit(1);
}

static inline void sleep_ns(uint64_t nanoseconds) {
    if (nanoseconds <= 0) return;

#ifdef _WIN32
    // Convert to milliseconds (round up to avoid undersleeping)
    DWORD ms = (DWORD)((nanoseconds + 999999ULL) / 1000000ULL);
    Sleep(ms);
#else
    struct timespec req, rem;

    req.tv_sec = nanoseconds / 1000000000LL;
    req.tv_nsec = nanoseconds % 1000000000LL;

    // Retry if interrupted by signal
    while (nanosleep(&req, &rem) == -1 && errno == EINTR) {
        req = rem;
    }
#endif
}
static inline void sleep_ms(uint64_t milliseconds) {
    sleep_ns(milliseconds * 1000000);
}

int main() {
    Result res;
    res = socket_init();
    if (res != SUCCESS) {
        fatal("Init failed", res);
    }
    printf("Socket library initialized\n");

#if 1
    socket_t socket;
    res = socket_connect_blocking("127.0.0.1", 9168, &socket);
    if (res != SUCCESS) {
        fatal("Connect failed", res);
    }
    printf("Connected\n");
    socket_close(socket);
#else
    TcpConnection connection = {0};
    while (true) {
        res = socket_connect("127.0.0.1", 9168, &connection);
        if (res == CONNECT_WAITING) {
            // Do nothing
        printf("Waiting for connection...\n");
        } else if (res == CONNECT_FAILED) {
            fatal("Connect failed", res);
        } else {
            break;
        }
        sleep_ms(10);
    }
    printf("Connected\n");
    socket_close_connection(&connection);
#endif
}

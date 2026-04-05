#define TRACING_IMPLEMENTATION
#include "tracing.h"

void fatal(const char* msg, Result result) {
    printf("%s: %d\n", msg, result);
    exit(1);
}

int main() {
    Result res;
    res = socket_init();
    if (res != SUCCESS) {
        fatal("Init failed", res);
    }
    printf("Socket library initialized\n");

    socket_t listening_socket;
    res = socket_listen(9168, 128, false, false, &listening_socket);
    if (res != SUCCESS) {
        fatal("Listen failed", res);
    }
    printf("Created listening socket\n");

    socket_t socket;
    res = socket_accept(listening_socket, -1, &socket);
    if (res != SUCCESS) {
        fatal("Accept failed", res);
    }
    printf("Accepted connection\n");

    socket_close(listening_socket);
    socket_close(socket);
}

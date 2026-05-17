#define TRACER_IMPLEMENTATION
#include "tracer.h"

extern "C" {
#include "testing.h"
}

// ============================================================
// Socket tests
// ============================================================

TEST(socket_init_deinit) {
    Result r = socket_init();
    CHECK(r == SUCCESS);
    socket_deinit();
}

TEST(listen_accept_connect) {
    socket_init();

    socket_t server_sock;
    Result r = socket_listen(0, 1, true, true, &server_sock);
    CHECK(r == SUCCESS);

    // Get the port the OS assigned
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    int rc = getsockname(server_sock, (struct sockaddr*)&addr, &addr_len);
    CHECK(rc == 0);
    uint16_t port = ntohs(addr.sin_port);
    CHECK(port != 0);

    // Connect from client
    socket_t client_sock;
    r = socket_connect_blocking("127.0.0.1", port, &client_sock);
    CHECK(r == SUCCESS);

    // Accept on server
    socket_t accepted_sock;
    r = socket_accept(server_sock, 1000, &accepted_sock);
    CHECK(r == SUCCESS);

    socket_close(accepted_sock);
    socket_close(client_sock);
    socket_close(server_sock);
    socket_deinit();
}

TEST(send_recv_data) {
    socket_init();

    socket_t server_sock;
    Result r = socket_listen(0, 1, true, true, &server_sock);
    CHECK(r == SUCCESS);

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    getsockname(server_sock, (struct sockaddr*)&addr, &addr_len);
    uint16_t port = ntohs(addr.sin_port);

    socket_t client_sock;
    r = socket_connect_blocking("127.0.0.1", port, &client_sock);
    CHECK(r == SUCCESS);

    socket_t accepted_sock;
    r = socket_accept(server_sock, 1000, &accepted_sock);
    CHECK(r == SUCCESS);

    // Client sends data
    const char* msg = "hello tracer";
    bool ok = socket_send_all(client_sock, msg, strlen(msg));
    CHECK(ok);

    // Server receives
    char buf[64] = {0};
    ssize_t n = recv(accepted_sock, buf, sizeof(buf), 0);
    CHECK(n == (ssize_t)strlen(msg));
    CHECK(memcmp(buf, msg, strlen(msg)) == 0);

    socket_close(accepted_sock);
    socket_close(client_sock);
    socket_close(server_sock);
    socket_deinit();
}

TEST(sendv_multiple_bufs) {
    socket_init();

    socket_t server_sock;
    socket_listen(0, 1, true, true, &server_sock);

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    getsockname(server_sock, (struct sockaddr*)&addr, &addr_len);
    uint16_t port = ntohs(addr.sin_port);

    socket_t client_sock;
    socket_connect_blocking("127.0.0.1", port, &client_sock);

    socket_t accepted_sock;
    socket_accept(server_sock, 1000, &accepted_sock);

    // Send via scatter/gather
    const char* part1 = "AAA";
    const char* part2 = "BBBBB";
    const char* part3 = "CC";
    SocketBuf bufs[3] = {
        SOCKET_BUF(part1, 3),
        SOCKET_BUF(part2, 5),
        SOCKET_BUF(part3, 2),
    };
    bool ok = socket_sendv(client_sock, bufs, 3);
    CHECK(ok);

    char buf[64] = {0};
    size_t total = 0;
    while (total < 10) {
        ssize_t n = recv(accepted_sock, buf + total, sizeof(buf) - total, 0);
        CHECK(n > 0);
        total += (size_t)n;
    }
    CHECK(memcmp(buf, "AAABBBBBCC", 10) == 0);

    socket_close(accepted_sock);
    socket_close(client_sock);
    socket_close(server_sock);
    socket_deinit();
}

TEST(connect_nonblocking) {
    socket_init();

    socket_t server_sock;
    socket_listen(0, 1, true, true, &server_sock);

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    getsockname(server_sock, (struct sockaddr*)&addr, &addr_len);
    uint16_t port = ntohs(addr.sin_port);

    TcpConnection conn;
    memset(&conn, 0, sizeof(conn));
    Result r = socket_connect("127.0.0.1", port, &conn);

    // Should be SUCCESS or CONNECT_WAITING
    CHECK(r == SUCCESS || r == CONNECT_WAITING);

    if (r == CONNECT_WAITING) {
        // Poll until connected
        for (int i = 0; i < 100; i++) {
            r = socket_connect("127.0.0.1", port, &conn);
            if (r == SUCCESS) break;
            CHECK(r == CONNECT_WAITING);
            thread_sleep_ms(10);
        }
        CHECK(r == SUCCESS);
    }

    socket_t accepted_sock;
    socket_accept(server_sock, 1000, &accepted_sock);

    // Verify the connection works
    bool ok = socket_send_all(conn.socket, "x", 1);
    CHECK(ok);

    char buf[4] = {0};
    ssize_t n = recv(accepted_sock, buf, sizeof(buf), 0);
    CHECK(n == 1);
    CHECK(buf[0] == 'x');

    socket_close(conn.socket);
    socket_close(accepted_sock);
    socket_close(server_sock);
    socket_deinit();
}

TEST(accept_timeout) {
    socket_init();

    socket_t server_sock;
    Result r = socket_listen(0, 1, true, true, &server_sock);
    CHECK(r == SUCCESS);

    socket_t client;
    r = socket_accept(server_sock, 50, &client);
    CHECK(r == TIMEOUT);

    socket_close(server_sock);
    socket_deinit();
}

TEST(connect_refused) {
    socket_init();

    // Port 1 is unlikely to have a listener
    socket_t sock;
    Result r = socket_connect_blocking("127.0.0.1", 1, &sock);
    CHECK(r == CONNECT_FAILED);

    socket_deinit();
}

// ============================================================
// main
// ============================================================

int main() {
    test_setup_timeout(30);

    printf("=== Socket Tests ===\n\n");

    RUN(socket_init_deinit);
    RUN(listen_accept_connect);
    RUN(send_recv_data);
    RUN(sendv_multiple_bufs);
    RUN(connect_nonblocking);
    RUN(accept_timeout);
    RUN(connect_refused);

    return test_print_results("Socket Tests");
}

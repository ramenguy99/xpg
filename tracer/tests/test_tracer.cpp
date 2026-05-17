#define TRACER_IMPLEMENTATION
#include "tracer.h"

extern "C" {
#include "testing.h"
}

// ============================================================
// Tracepoint registration tests
// ============================================================

TRACEPOINT_DEFINE(tp_test_a, "test.alpha");
TRACEPOINT_DEFINE(tp_test_b, "test.beta");
TRACEPOINT_DEFINE(tp_test_c, "test.gamma");

TEST(tracepoints_registered) {
    CHECK(tp_test_a != NULL);
    CHECK(tp_test_b != NULL);
    CHECK(tp_test_c != NULL);
    CHECK(strcmp(tp_test_a->name, "test.alpha") == 0);
    CHECK(strcmp(tp_test_b->name, "test.beta") == 0);
    CHECK(strcmp(tp_test_c->name, "test.gamma") == 0);
}

TEST(tracepoint_initially_disabled) {
    tracer_init();
    CHECK(!tracepoint_enabled(tp_test_a));
    CHECK(!tracepoint_enabled(tp_test_b));
    tracer_close();
}

// ============================================================
// TCP subscriber tests
// ============================================================

struct ServerData {
    socket_t listen_sock;
    socket_t client_sock;
    uint16_t port;
    _Atomic(int) ready;
    _Atomic(int) connected;
    uint8_t recv_buf[4096];
    size_t recv_len;
};

static THREAD_PROC(mock_server_thread) {
    ServerData* s = (ServerData*)data;

    socket_init();
    Result r = socket_listen(0, 1, true, true, &s->listen_sock);
    if (r != SUCCESS) return 0;

    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    getsockname(s->listen_sock, (struct sockaddr*)&addr, &addr_len);
    s->port = ntohs(addr.sin_port);
    atomic_store_explicit(&s->ready, 1, memory_order_release);

    r = socket_accept(s->listen_sock, 5000, &s->client_sock);
    if (r != SUCCESS) {
        socket_close(s->listen_sock);
        return 0;
    }
    atomic_store_explicit(&s->connected, 1, memory_order_release);

    // Read data until connection closes or buffer full
    s->recv_len = 0;
    while (s->recv_len < sizeof(s->recv_buf)) {
        ssize_t n = recv(s->client_sock, (char*)(s->recv_buf + s->recv_len),
                        (int)(sizeof(s->recv_buf) - s->recv_len), 0);
        if (n <= 0) break;
        s->recv_len += (size_t)n;
    }

    return 0;
}

TEST(tcp_subscriber_connects_and_sends) {
    ServerData server;
    memset(&server, 0, sizeof(server));
    atomic_store_explicit(&server.ready, 0, memory_order_relaxed);
    atomic_store_explicit(&server.connected, 0, memory_order_relaxed);

    Thread server_thread;
    create_thread(mock_server_thread, &server, &server_thread);

    // Wait for server to be ready
    while (!atomic_load_explicit(&server.ready, memory_order_acquire))
        thread_sleep_ms(1);

    tracer_init();

    int idx = tracer_add_tcp_subscriber("127.0.0.1", server.port);
    CHECK(idx >= 0);
    tracer_subscribe_all(idx);

    // Wait for connection
    for (int i = 0; i < 200; i++) {
        if (atomic_load_explicit(&server.connected, memory_order_acquire)) break;
        thread_sleep_ms(10);
    }
    CHECK(atomic_load_explicit(&server.connected, memory_order_acquire));

    // Emit a trace event
    TRACE(tp_test_a,
        TI64("value", 42),
        TSTR("msg", "hello", 5)
    );

    // Give the consumer thread time to send
    thread_sleep_ms(100);

    tracer_close();

    // Server thread should finish once connection closes
    join_thread(&server_thread);
    socket_close(server.client_sock);
    socket_close(server.listen_sock);
    socket_deinit();

    // Verify we received something
    CHECK(server.recv_len > 0);

    // Check handshake: "AMBR" magic
    CHECK(server.recv_len >= 14);
    CHECK(memcmp(server.recv_buf, "AMBR", 4) == 0);

    // After handshake (14 bytes), there should be a trace event TLV
    CHECK(server.recv_len > 14);
    uint8_t* tlv = server.recv_buf + 14;
    uint32_t msg_type;
    memcpy(&msg_type, tlv, 4);
    CHECK_EQ(msg_type, MSG_TRACE_EVENT);
}

TEST(tcp_subscribe_unsubscribe) {
    ServerData server;
    memset(&server, 0, sizeof(server));
    atomic_store_explicit(&server.ready, 0, memory_order_relaxed);
    atomic_store_explicit(&server.connected, 0, memory_order_relaxed);

    Thread server_thread;
    create_thread(mock_server_thread, &server, &server_thread);

    while (!atomic_load_explicit(&server.ready, memory_order_acquire))
        thread_sleep_ms(1);

    tracer_init();

    int idx = tracer_add_tcp_subscriber("127.0.0.1", server.port);
    CHECK(idx >= 0);

    // Subscribe only to test.alpha
    CHECK(tracer_subscribe(idx, "test.alpha"));
    CHECK(tracepoint_enabled(tp_test_a));
    CHECK(!tracepoint_enabled(tp_test_b));

    // Unsubscribe
    CHECK(tracer_unsubscribe(idx, "test.alpha"));
    CHECK(!tracepoint_enabled(tp_test_a));

    // Subscribe all
    tracer_subscribe_all(idx);
    CHECK(tracepoint_enabled(tp_test_a));
    CHECK(tracepoint_enabled(tp_test_b));
    CHECK(tracepoint_enabled(tp_test_c));

    // Unsubscribe all
    tracer_unsubscribe_all(idx);
    CHECK(!tracepoint_enabled(tp_test_a));
    CHECK(!tracepoint_enabled(tp_test_b));

    tracer_close();
    join_thread(&server_thread);
    socket_close(server.client_sock);
    socket_close(server.listen_sock);
    socket_deinit();
}

TEST(tcp_multiple_events) {
    ServerData server;
    memset(&server, 0, sizeof(server));
    atomic_store_explicit(&server.ready, 0, memory_order_relaxed);
    atomic_store_explicit(&server.connected, 0, memory_order_relaxed);

    Thread server_thread;
    create_thread(mock_server_thread, &server, &server_thread);

    while (!atomic_load_explicit(&server.ready, memory_order_acquire))
        thread_sleep_ms(1);

    tracer_init();

    int idx = tracer_add_tcp_subscriber("127.0.0.1", server.port);
    tracer_subscribe_all(idx);

    for (int i = 0; i < 100; i++) {
        if (atomic_load_explicit(&server.connected, memory_order_acquire)) break;
        thread_sleep_ms(10);
    }

    // Emit multiple events
    for (int i = 0; i < 10; i++) {
        TRACE(tp_test_b, TI64("i", i));
    }

    thread_sleep_ms(100);
    tracer_close();

    join_thread(&server_thread);
    socket_close(server.client_sock);
    socket_close(server.listen_sock);
    socket_deinit();

    // Should have handshake + 10 TLV messages
    // Each TLV has at minimum 16 bytes header
    CHECK(server.recv_len > 14 + 10 * 16);
}

// ============================================================
// SQLite subscriber tests (conditional)
// ============================================================

#ifdef TRACER_SQLITE_ENABLED

TEST(sqlite_subscriber_writes_events) {
    const char* db_path = "/tmp/test_tracer_sqlite.db";
    remove(db_path);

    tracer_init();

    SqliteConfig cfg = sqlite_config_default();
    int idx = tracer_add_sqlite_subscriber(db_path, &cfg);
    CHECK(idx >= 0);
    tracer_subscribe_all(idx);

    // Emit events
    for (int i = 0; i < 5; i++) {
        TRACE(tp_test_a,
            TI64("value", i),
            TF64("metric", i * 1.5)
        );
    }

    thread_sleep_ms(200);
    tracer_close();

    // Verify the database has data
    sqlite3* db;
    int rc = sqlite3_open(db_path, &db);
    CHECK(rc == SQLITE_OK);

    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM \"test.alpha\"", -1, &stmt, NULL);
    CHECK(rc == SQLITE_OK);
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW);
    int count = sqlite3_column_int(stmt, 0);
    CHECK_EQ(count, 5);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    remove(db_path);
}

TEST(sqlite_subscriber_correct_values) {
    const char* db_path = "/tmp/test_tracer_sqlite_vals.db";
    remove(db_path);

    tracer_init();

    SqliteConfig cfg = sqlite_config_default();
    int idx = tracer_add_sqlite_subscriber(db_path, &cfg);
    tracer_subscribe_all(idx);

    TRACE(tp_test_c,
        TI64("x", 123),
        TF64("y", 4.5),
        TSTR("name", "test", 4)
    );

    thread_sleep_ms(200);
    tracer_close();

    sqlite3* db;
    sqlite3_open(db_path, &db);

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, "SELECT \"x\", \"y\", \"name\" FROM \"test.gamma\"", -1, &stmt, NULL);
    int rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW);

    CHECK_EQ(sqlite3_column_int64(stmt, 0), 123);
    CHECK(sqlite3_column_double(stmt, 1) > 4.49 && sqlite3_column_double(stmt, 1) < 4.51);
    const char* name = (const char*)sqlite3_column_text(stmt, 2);
    CHECK(name != NULL);
    CHECK(memcmp(name, "test", 4) == 0);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    remove(db_path);
}

#endif // TRACER_SQLITE_ENABLED

// ============================================================
// main
// ============================================================

int main() {
    test_setup_timeout(30);

    printf("=== Tracer Tests ===\n\n");

    printf("Registration:\n");
    RUN(tracepoints_registered);
    RUN(tracepoint_initially_disabled);

    printf("\nTCP subscriber:\n");
    RUN(tcp_subscriber_connects_and_sends);
    RUN(tcp_subscribe_unsubscribe);
    RUN(tcp_multiple_events);

#ifdef TRACER_SQLITE_ENABLED
    printf("\nSQLite subscriber:\n");
    RUN(sqlite_subscriber_writes_events);
    RUN(sqlite_subscriber_correct_values);
#else
    printf("\n(SQLite tests skipped - TRACER_SQLITE_ENABLED not defined)\n");
#endif

    return test_print_results("Tracer Tests");
}

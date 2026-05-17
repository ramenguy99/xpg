#define TRACER_QUEUE_FULL_POLICY 1
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
    atomic_uint ready;
    atomic_uint connected;
    uint8_t recv_buf[16384];
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
// List serialization tests
// ============================================================

static uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static int64_t read_i64_le(const uint8_t* p) {
    int64_t v;
    memcpy(&v, p, 8);
    return v;
}

static double read_f64_le(const uint8_t* p) {
    double v;
    memcpy(&v, p, 8);
    return v;
}

TEST(list_size_empty) {
    TraceField items[1]; // unused
    (void)items;
    // An empty list: key + type(8) + count(8)
    size_t key_len = 4; // "test"
    size_t expected = (8 + align_up(key_len, 8)) + 8 + 8;
    size_t got = trace_size_list(key_len, NULL, 0);
    CHECK_EQ(got, expected);
}

TEST(list_size_ints) {
    TraceField items[] = { TI64("", 1), TI64("", 2), TI64("", 3) };
    size_t key_len = 4;
    // key_size + type(8) + count(8) + 3*(type(8)+val(8))
    size_t expected = (8 + align_up(key_len, 8)) + 8 + 8 + 3 * (8 + 8);
    size_t got = trace_size_list(key_len, items, 3);
    CHECK_EQ(got, expected);
}

TEST(list_write_ints) {
    TraceField items[] = { TI64("", 10), TI64("", 20), TI64("", 30) };
    TraceField f = TLIST("key", items, 3);

    uint8_t buf[256];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    // Key: key_len=3, "key", padded to 8
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    CHECK(memcmp(p, "key", 3) == 0); p += 8; // padded

    // Type code = LIST (4)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    // Count = 3
    CHECK_EQ(read_u64_le(p), 3); p += 8;

    // Element 0: type=I64, val=10
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 10); p += 8;
    // Element 1: type=I64, val=20
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 20); p += 8;
    // Element 2: type=I64, val=30
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 30); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

TEST(list_write_mixed_types) {
    TraceField items[] = {
        TI64("", 42),
        TF64("", 3.14),
        TSTR("", "hi", 2),
        TNONE("")
    };
    TraceField f = TLIST("mix", items, 4);

    uint8_t buf[512];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    // Key: "mix" (len=3)
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    CHECK(memcmp(p, "mix", 3) == 0); p += 8;

    // Type = LIST
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    // Count = 4
    CHECK_EQ(read_u64_le(p), 4); p += 8;

    // Element 0: I64 = 42
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 42); p += 8;

    // Element 1: F64 = 3.14
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_F64); p += 8;
    CHECK(read_f64_le(p) > 3.13 && read_f64_le(p) < 3.15); p += 8;

    // Element 2: STR = "hi" (len=2, padded to 8)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;
    CHECK(memcmp(p, "hi", 2) == 0); p += 8; // padded to 8

    // Element 3: NONE
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_NONE); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

TEST(list_write_nested) {
    TraceField inner[] = { TI64("", 100), TI64("", 200) };
    TraceField outer[] = {
        TI64("", 1),
        TLIST("", inner, 2)
    };
    TraceField f = TLIST("nest", outer, 2);

    uint8_t buf[512];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    // Key: "nest" (len=4)
    CHECK_EQ(read_u64_le(p), 4); p += 8;
    CHECK(memcmp(p, "nest", 4) == 0); p += 8;

    // Outer list: type=LIST, count=2
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;

    // Outer[0]: I64 = 1
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 1); p += 8;

    // Outer[1]: nested LIST, count=2
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;

    // Inner[0]: I64 = 100
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 100); p += 8;
    // Inner[1]: I64 = 200
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 200); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

TEST(list_tcp_end_to_end) {
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
    CHECK(atomic_load_explicit(&server.connected, memory_order_acquire));

    // Emit a trace with a list field
    TraceField items[] = { TI64("", 1), TI64("", 2), TI64("", 3) };
    TRACE(tp_test_a,
        TSTR("tag", "list_test", 9),
        TLIST("data", items, 3)
    );

    thread_sleep_ms(100);
    tracer_close();

    join_thread(&server_thread);
    socket_close(server.client_sock);
    socket_close(server.listen_sock);
    socket_deinit();

    // Verify handshake + at least one TLV arrived
    CHECK(server.recv_len > 14);
    CHECK(memcmp(server.recv_buf, "AMBR", 4) == 0);

    // Parse the TLV after handshake
    uint8_t* tlv = server.recv_buf + 14;
    uint32_t msg_type;
    memcpy(&msg_type, tlv, 4);
    CHECK_EQ(msg_type, MSG_TRACE_EVENT);

    // Get payload length
    uint64_t payload_len;
    memcpy(&payload_len, tlv + 8, 8);
    CHECK(payload_len > 0);

    // The payload contains: id_len(8) + name + num_entries(8) + fields
    uint8_t* payload = tlv + 16;
    uint64_t id_len = read_u64_le(payload);
    CHECK_EQ(id_len, strlen("test.alpha"));
    payload += 8 + id_len;

    // num_entries = 2 (tag + data)
    uint64_t num_entries = read_u64_le(payload);
    CHECK_EQ(num_entries, 2);
    payload += 8;

    // Skip first field (tag: STR)
    uint64_t key_len = read_u64_le(payload); payload += 8;
    payload += align_up(key_len, 8); // key padded
    uint64_t type_code = read_u64_le(payload); payload += 8;
    CHECK_EQ(type_code, TRACE_TYPE_STR);
    uint64_t str_len = read_u64_le(payload); payload += 8;
    payload += align_up(str_len, 8);

    // Second field: "data" list
    key_len = read_u64_le(payload); payload += 8;
    CHECK_EQ(key_len, 4);
    CHECK(memcmp(payload, "data", 4) == 0);
    payload += align_up(key_len, 8);

    type_code = read_u64_le(payload); payload += 8;
    CHECK_EQ(type_code, TRACE_TYPE_LIST);

    uint64_t count = read_u64_le(payload); payload += 8;
    CHECK_EQ(count, 3);

    // Elements: 1, 2, 3
    for (int i = 0; i < 3; i++) {
        CHECK_EQ(read_u64_le(payload), (uint64_t)TRACE_TYPE_I64); payload += 8;
        CHECK_EQ(read_i64_le(payload), i + 1); payload += 8;
    }
}

// ============================================================
// Tuple serialization tests
// ============================================================

TEST(tuple_size_empty) {
    size_t key_len = 3; // "tup"
    size_t expected = (8 + align_up(key_len, 8)) + 8 + 8;
    size_t got = trace_size_tuple(key_len, NULL, 0);
    CHECK_EQ(got, expected);
}

TEST(tuple_write_ints) {
    TraceField items[] = { TI64("", 10), TI64("", 20), TI64("", 30) };
    TraceField f = TTUPLE("key", items, 3);

    uint8_t buf[256];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    CHECK(memcmp(p, "key", 3) == 0); p += 8;

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_TUPLE); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 10); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 20); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 30); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

TEST(tuple_write_mixed) {
    TraceField items[] = {
        TI64("", 99),
        TSTR("", "abc", 3),
        TF64("", 2.5)
    };
    TraceField f = TTUPLE("t", items, 3);

    uint8_t buf[256];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    // Key: "t" (len=1)
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "t", 1) == 0); p += 8;

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_TUPLE); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;

    // I64 = 99
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 99); p += 8;

    // STR = "abc"
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    CHECK(memcmp(p, "abc", 3) == 0); p += 8;

    // F64 = 2.5
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_F64); p += 8;
    CHECK(read_f64_le(p) > 2.49 && read_f64_le(p) < 2.51); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

// ============================================================
// Dict serialization tests
// ============================================================

TEST(dict_size_empty) {
    size_t key_len = 1; // "d"
    size_t expected = (8 + align_up(key_len, 8)) + 8 + 8;
    size_t got = trace_size_dict(key_len, NULL, 0);
    CHECK_EQ(got, expected);
}

TEST(dict_write_int_values) {
    // pairs: key0, val0, key1, val1
    TraceField pairs[] = {
        TSTR("", "x", 1), TI64("", 10),
        TSTR("", "y", 1), TI64("", 20),
    };
    TraceField f = TDICT("d", pairs, 2);

    uint8_t buf[512];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    // Key: "d" (len=1)
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "d", 1) == 0); p += 8;

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;

    // Pair 0: STR "x" -> I64 10
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "x", 1) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 10); p += 8;

    // Pair 1: STR "y" -> I64 20
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "y", 1) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 20); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

TEST(dict_write_mixed_values) {
    TraceField pairs[] = {
        TI64("", 1), TSTR("", "one", 3),
        TI64("", 2), TF64("", 2.0),
        TI64("", 3), TNONE(""),
    };
    TraceField f = TDICT("map", pairs, 3);

    uint8_t buf[512];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    // Key: "map" (len=3)
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    CHECK(memcmp(p, "map", 3) == 0); p += 8;

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;

    // Pair 0: I64(1) -> STR("one")
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 1); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    CHECK(memcmp(p, "one", 3) == 0); p += 8;

    // Pair 1: I64(2) -> F64(2.0)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 2); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_F64); p += 8;
    CHECK(read_f64_le(p) > 1.99 && read_f64_le(p) < 2.01); p += 8;

    // Pair 2: I64(3) -> NONE
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 3); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_NONE); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

TEST(dict_nested_in_list) {
    TraceField pairs[] = {
        TSTR("", "a", 1), TI64("", 1),
    };
    TraceField list_items[] = {
        TI64("", 42),
        TDICT("", pairs, 1),
    };
    TraceField f = TLIST("data", list_items, 2);

    uint8_t buf[512];
    memset(buf, 0, sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, _trace_field_size(&f));

    uint8_t* p = buf;
    // Key: "data" (len=4)
    CHECK_EQ(read_u64_le(p), 4); p += 8;
    CHECK(memcmp(p, "data", 4) == 0); p += 8;

    // LIST with count=2
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;

    // Element 0: I64 = 42
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 42); p += 8;

    // Element 1: DICT with count=1
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;

    // Pair: STR("a") -> I64(1)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "a", 1) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 1); p += 8;

    CHECK_EQ((size_t)(p - buf), written);
}

TEST(tuple_and_dict_tcp_end_to_end) {
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
    CHECK(atomic_load_explicit(&server.connected, memory_order_acquire));

    // Emit a trace with tuple and dict fields
    TraceField tup_items[] = { TI64("", 1), TI64("", 2) };
    TraceField dict_pairs[] = {
        TSTR("", "k", 1), TI64("", 99),
    };
    TRACE(tp_test_a,
        TTUPLE("pos", tup_items, 2),
        TDICT("meta", dict_pairs, 1)
    );

    thread_sleep_ms(100);
    tracer_close();

    join_thread(&server_thread);
    socket_close(server.client_sock);
    socket_close(server.listen_sock);
    socket_deinit();

    CHECK(server.recv_len > 14);
    CHECK(memcmp(server.recv_buf, "AMBR", 4) == 0);

    uint8_t* tlv = server.recv_buf + 14;
    uint32_t msg_type;
    memcpy(&msg_type, tlv, 4);
    CHECK_EQ(msg_type, MSG_TRACE_EVENT);

    uint64_t payload_len;
    memcpy(&payload_len, tlv + 8, 8);
    CHECK(payload_len > 0);

    // Parse payload: id + num_entries
    uint8_t* payload = tlv + 16;
    uint64_t id_len = read_u64_le(payload);
    CHECK_EQ(id_len, strlen("test.alpha"));
    payload += 8 + id_len;

    uint64_t num_entries = read_u64_le(payload);
    CHECK_EQ(num_entries, 2);
    payload += 8;

    // First field: "pos" tuple
    uint64_t key_len = read_u64_le(payload); payload += 8;
    CHECK_EQ(key_len, 3);
    CHECK(memcmp(payload, "pos", 3) == 0);
    payload += align_up(key_len, 8);
    uint64_t type_code = read_u64_le(payload); payload += 8;
    CHECK_EQ(type_code, TRACE_TYPE_TUPLE);
    uint64_t count = read_u64_le(payload); payload += 8;
    CHECK_EQ(count, 2);
    // Skip tuple elements
    for (int i = 0; i < 2; i++) {
        payload += 8; // type
        payload += 8; // value
    }

    // Second field: "meta" dict
    key_len = read_u64_le(payload); payload += 8;
    CHECK_EQ(key_len, 4);
    CHECK(memcmp(payload, "meta", 4) == 0);
    payload += align_up(key_len, 8);
    type_code = read_u64_le(payload); payload += 8;
    CHECK_EQ(type_code, TRACE_TYPE_DICT);
    count = read_u64_le(payload); payload += 8;
    CHECK_EQ(count, 1);
}

// ============================================================
// Nested structure stress test
// ============================================================

TEST(nested_stress) {
    // Build a deeply nested structure:
    // DICT {
    //   "config" -> TUPLE(I64, F64, STR),
    //   "items"  -> LIST[
    //     DICT{ "id"->I64, "tags"->LIST[STR, STR] },
    //     DICT{ "id"->I64, "tags"->LIST[STR] },
    //     TUPLE(NONE, I64, DICT{ "x"->F64 })
    //   ],
    //   "empty"  -> LIST[],
    //   "deep"   -> LIST[ TUPLE( LIST[ DICT{ "v"->I64 } ] ) ]
    // }

    // "config" -> TUPLE(I64, F64, STR)
    TraceField config_items[] = {
        TI64("", 42),
        TF64("", 3.14),
        TSTR("", "hello", 5),
    };

    // items[0]: DICT{ "id"->I64(1), "tags"->LIST["a","b"] }
    TraceField tags0[] = { TSTR("", "a", 1), TSTR("", "b", 1) };
    TraceField item0_pairs[] = {
        TSTR("", "id", 2), TI64("", 1),
        TSTR("", "tags", 4), TLIST("", tags0, 2),
    };

    // items[1]: DICT{ "id"->I64(2), "tags"->LIST["c"] }
    TraceField tags1[] = { TSTR("", "c", 1) };
    TraceField item1_pairs[] = {
        TSTR("", "id", 2), TI64("", 2),
        TSTR("", "tags", 4), TLIST("", tags1, 1),
    };

    // items[2]: TUPLE(NONE, I64(99), DICT{ "x"->F64(1.5) })
    TraceField item2_dict_pairs[] = {
        TSTR("", "x", 1), TF64("", 1.5),
    };
    TraceField item2_tuple_items[] = {
        TNONE(""),
        TI64("", 99),
        TDICT("", item2_dict_pairs, 1),
    };

    // items list
    TraceField items[] = {
        TDICT("", item0_pairs, 2),
        TDICT("", item1_pairs, 2),
        TTUPLE("", item2_tuple_items, 3),
    };

    // "deep" -> LIST[ TUPLE( LIST[ DICT{ "v"->I64(777) } ] ) ]
    TraceField deep_dict_pairs[] = {
        TSTR("", "v", 1), TI64("", 777),
    };
    TraceField deep_list_inner[] = { TDICT("", deep_dict_pairs, 1) };
    TraceField deep_tuple_items[] = { TLIST("", deep_list_inner, 1) };
    TraceField deep_list[] = { TTUPLE("", deep_tuple_items, 1) };

    // Top-level dict: 4 pairs
    TraceField top_pairs[] = {
        TSTR("", "config", 6), TTUPLE("", config_items, 3),
        TSTR("", "items", 5),  TLIST("", items, 3),
        TSTR("", "empty", 5),  TLIST("", NULL, 0),
        TSTR("", "deep", 4),   TLIST("", deep_list, 1),
    };
    TraceField f = TDICT("root", top_pairs, 4);

    // Serialize
    uint8_t buf[4096];
    memset(buf, 0, sizeof(buf));
    size_t expected_size = _trace_field_size(&f);
    CHECK(expected_size > 0);
    CHECK(expected_size < sizeof(buf));
    uint8_t* end = _trace_field_write(buf, &f);
    size_t written = (size_t)(end - buf);
    CHECK_EQ(written, expected_size);

    // Parse and verify the top-level structure
    uint8_t* p = buf;

    // Key: "root" (len=4)
    CHECK_EQ(read_u64_le(p), 4); p += 8;
    CHECK(memcmp(p, "root", 4) == 0); p += 8;

    // DICT type, count=4
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 4); p += 8;

    // --- Pair 0: "config" -> TUPLE(3) ---
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 6); p += 8;
    CHECK(memcmp(p, "config", 6) == 0); p += 8; // padded to 8

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_TUPLE); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    // tuple[0]: I64(42)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 42); p += 8;
    // tuple[1]: F64(3.14)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_F64); p += 8;
    CHECK(read_f64_le(p) > 3.13 && read_f64_le(p) < 3.15); p += 8;
    // tuple[2]: STR("hello")
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 5); p += 8;
    CHECK(memcmp(p, "hello", 5) == 0); p += 8;

    // --- Pair 1: "items" -> LIST(3) ---
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 5); p += 8;
    CHECK(memcmp(p, "items", 5) == 0); p += 8;

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;

    // items[0]: DICT(2)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;
    // pair "id"->1
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;
    CHECK(memcmp(p, "id", 2) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 1); p += 8;
    // pair "tags"->LIST(2)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 4); p += 8;
    CHECK(memcmp(p, "tags", 4) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;
    // tags: "a", "b"
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "a", 1) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "b", 1) == 0); p += 8;

    // items[1]: DICT(2)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;
    // pair "id"->2
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;
    CHECK(memcmp(p, "id", 2) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 2); p += 8;
    // pair "tags"->LIST(1)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 4); p += 8;
    CHECK(memcmp(p, "tags", 4) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "c", 1) == 0); p += 8;

    // items[2]: TUPLE(3): NONE, I64(99), DICT(1)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_TUPLE); p += 8;
    CHECK_EQ(read_u64_le(p), 3); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_NONE); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 99); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    // dict pair: "x"->F64(1.5)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "x", 1) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_F64); p += 8;
    CHECK(read_f64_le(p) > 1.49 && read_f64_le(p) < 1.51); p += 8;

    // --- Pair 2: "empty" -> LIST(0) ---
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 5); p += 8;
    CHECK(memcmp(p, "empty", 5) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 0); p += 8;

    // --- Pair 3: "deep" -> LIST(1) containing TUPLE(1) containing LIST(1) containing DICT(1) ---
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 4); p += 8;
    CHECK(memcmp(p, "deep", 4) == 0); p += 8;

    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    // -> TUPLE(1)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_TUPLE); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    // -> LIST(1)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    // -> DICT(1)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    // pair "v"->I64(777)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "v", 1) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 777); p += 8;

    // Verify we consumed exactly what was written
    CHECK_EQ((size_t)(p - buf), written);
}

TEST(nested_stress_repeated_serialize) {
    // Verify that serializing the same structure many times produces identical output
    TraceField inner_pairs[] = {
        TI64("", 0), TSTR("", "val", 3),
    };
    TraceField tuple_items[] = {
        TF64("", -1.0),
        TDICT("", inner_pairs, 1),
        TLIST("", NULL, 0),
    };
    TraceField list_items[] = {
        TTUPLE("", tuple_items, 3),
        TTUPLE("", tuple_items, 3),
        TI64("", 999),
    };
    TraceField f = TLIST("rep", list_items, 3);

    uint8_t buf1[2048];
    uint8_t buf2[2048];
    memset(buf1, 0, sizeof(buf1));
    memset(buf2, 0xCC, sizeof(buf2));

    size_t size = _trace_field_size(&f);
    CHECK(size > 0);
    CHECK(size < sizeof(buf1));

    uint8_t* end1 = _trace_field_write(buf1, &f);
    uint8_t* end2 = _trace_field_write(buf2, &f);

    size_t written1 = (size_t)(end1 - buf1);
    size_t written2 = (size_t)(end2 - buf2);
    CHECK_EQ(written1, size);
    CHECK_EQ(written2, size);
    CHECK(memcmp(buf1, buf2, size) == 0);

    // Serialize 100 more times and verify consistency
    for (int i = 0; i < 100; i++) {
        memset(buf2, (uint8_t)i, sizeof(buf2));
        uint8_t* e = _trace_field_write(buf2, &f);
        CHECK_EQ((size_t)(e - buf2), size);
        CHECK(memcmp(buf1, buf2, size) == 0);
    }
}

TEST(nested_stress_tcp) {
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
    CHECK(atomic_load_explicit(&server.connected, memory_order_acquire));

    // Emit many events with complex nested structures
    for (int i = 0; i < 20; i++) {
        TraceField dict_pairs[] = {
            TI64("", i), TF64("", i * 0.5),
        };
        TraceField tuple_items[] = {
            TI64("", i),
            TSTR("", "x", 1),
            TDICT("", dict_pairs, 1),
        };
        TraceField list_items[] = {
            TTUPLE("", tuple_items, 3),
            TNONE(""),
        };
        TRACE(tp_test_b,
            TLIST("data", list_items, 2),
            TI64("seq", i)
        );
    }

    thread_sleep_ms(200);
    tracer_close();

    join_thread(&server_thread);
    socket_close(server.client_sock);
    socket_close(server.listen_sock);
    socket_deinit();

    // Verify handshake + all 20 messages arrived
    CHECK(server.recv_len > 14);
    CHECK(memcmp(server.recv_buf, "AMBR", 4) == 0);

    // Count TLV messages after handshake
    uint8_t* pos = server.recv_buf + 14;
    uint8_t* end = server.recv_buf + server.recv_len;
    int msg_count = 0;
    while (pos + 16 <= end) {
        uint32_t mt;
        memcpy(&mt, pos, 4);
        CHECK_EQ(mt, MSG_TRACE_EVENT);
        uint64_t plen;
        memcpy(&plen, pos + 8, 8);
        pos += 16 + plen;
        msg_count++;
    }
    CHECK_EQ(msg_count, 20);
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

TEST(sqlite_list_tuple_dict) {
    const char* db_path = "/tmp/test_tracer_sqlite_containers.db";
    remove(db_path);

    tracer_init();

    SqliteConfig cfg = sqlite_config_default();
    int idx = tracer_add_sqlite_subscriber(db_path, &cfg);
    tracer_subscribe_all(idx);

    // Emit a trace with list, tuple, dict fields
    TraceField list_items[] = { TI64("", 10), TI64("", 20), TI64("", 30) };
    TraceField tuple_items[] = { TF64("", 1.5), TSTR("", "hi", 2) };
    TraceField dict_pairs[] = {
        TSTR("", "key", 3), TI64("", 42),
        TSTR("", "val", 3), TF64("", 9.9),
    };
    TRACE(tp_test_a,
        TI64("id", 1),
        TLIST("nums", list_items, 3),
        TTUPLE("pos", tuple_items, 2),
        TDICT("meta", dict_pairs, 2)
    );

    thread_sleep_ms(200);
    tracer_close();

    sqlite3* db;
    int rc = sqlite3_open(db_path, &db);
    CHECK(rc == SQLITE_OK);

    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db,
        "SELECT \"id\", \"nums\", \"pos\", \"meta\" FROM \"test.alpha\"",
        -1, &stmt, NULL);
    CHECK(rc == SQLITE_OK);
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW);

    // id should be integer
    CHECK_EQ(sqlite3_column_int64(stmt, 0), 1);

    // nums should be a BLOB with LIST type code
    CHECK(sqlite3_column_type(stmt, 1) == SQLITE_BLOB);
    const uint8_t* nums_blob = (const uint8_t*)sqlite3_column_blob(stmt, 1);
    int nums_len = sqlite3_column_bytes(stmt, 1);
    CHECK(nums_blob != NULL);
    CHECK(nums_len > 0);
    // First 8 bytes: type code = LIST(4)
    CHECK_EQ(read_u64_le(nums_blob), TRACE_TYPE_LIST);
    // Next 8 bytes: count = 3
    CHECK_EQ(read_u64_le(nums_blob + 8), 3);
    // Elements: I64(10), I64(20), I64(30)
    const uint8_t* ep = nums_blob + 16;
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_I64); ep += 8;
    CHECK_EQ(read_i64_le(ep), 10); ep += 8;
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_I64); ep += 8;
    CHECK_EQ(read_i64_le(ep), 20); ep += 8;
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_I64); ep += 8;
    CHECK_EQ(read_i64_le(ep), 30); ep += 8;
    CHECK_EQ((int)(ep - nums_blob), nums_len);

    // pos should be a BLOB with TUPLE type code
    CHECK(sqlite3_column_type(stmt, 2) == SQLITE_BLOB);
    const uint8_t* pos_blob = (const uint8_t*)sqlite3_column_blob(stmt, 2);
    int pos_len = sqlite3_column_bytes(stmt, 2);
    CHECK(pos_blob != NULL);
    CHECK_EQ(read_u64_le(pos_blob), TRACE_TYPE_TUPLE);
    CHECK_EQ(read_u64_le(pos_blob + 8), 2);
    ep = pos_blob + 16;
    // F64(1.5)
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_F64); ep += 8;
    CHECK(read_f64_le(ep) > 1.49 && read_f64_le(ep) < 1.51); ep += 8;
    // STR("hi", len=2)
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_STR); ep += 8;
    CHECK_EQ(read_u64_le(ep), 2); ep += 8;
    CHECK(memcmp(ep, "hi", 2) == 0); ep += 8; // padded to 8
    CHECK_EQ((int)(ep - pos_blob), pos_len);

    // meta should be a BLOB with DICT type code
    CHECK(sqlite3_column_type(stmt, 3) == SQLITE_BLOB);
    const uint8_t* meta_blob = (const uint8_t*)sqlite3_column_blob(stmt, 3);
    int meta_len = sqlite3_column_bytes(stmt, 3);
    CHECK(meta_blob != NULL);
    CHECK_EQ(read_u64_le(meta_blob), TRACE_TYPE_DICT);
    CHECK_EQ(read_u64_le(meta_blob + 8), 2);
    ep = meta_blob + 16;
    // Pair 0: STR("key") -> I64(42)
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_STR); ep += 8;
    CHECK_EQ(read_u64_le(ep), 3); ep += 8;
    CHECK(memcmp(ep, "key", 3) == 0); ep += 8;
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_I64); ep += 8;
    CHECK_EQ(read_i64_le(ep), 42); ep += 8;
    // Pair 1: STR("val") -> F64(9.9)
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_STR); ep += 8;
    CHECK_EQ(read_u64_le(ep), 3); ep += 8;
    CHECK(memcmp(ep, "val", 3) == 0); ep += 8;
    CHECK_EQ(read_u64_le(ep), TRACE_TYPE_F64); ep += 8;
    CHECK(read_f64_le(ep) > 9.89 && read_f64_le(ep) < 9.91); ep += 8;
    CHECK_EQ((int)(ep - meta_blob), meta_len);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    remove(db_path);
}

TEST(sqlite_nested_containers) {
    const char* db_path = "/tmp/test_tracer_sqlite_nested.db";
    remove(db_path);

    tracer_init();

    SqliteConfig cfg = sqlite_config_default();
    int idx = tracer_add_sqlite_subscriber(db_path, &cfg);
    tracer_subscribe_all(idx);

    // Nested: LIST[ TUPLE(I64, DICT{ STR->F64 }), NONE ]
    TraceField inner_dict_pairs[] = {
        TSTR("", "x", 1), TF64("", 3.14),
    };
    TraceField inner_tuple[] = {
        TI64("", 7),
        TDICT("", inner_dict_pairs, 1),
    };
    TraceField outer_list[] = {
        TTUPLE("", inner_tuple, 2),
        TNONE(""),
    };

    TRACE(tp_test_b,
        TLIST("nested", outer_list, 2)
    );

    thread_sleep_ms(200);
    tracer_close();

    sqlite3* db;
    sqlite3_open(db_path, &db);

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db,
        "SELECT \"nested\" FROM \"test.beta\"", -1, &stmt, NULL);
    CHECK(rc == SQLITE_OK);
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW);

    CHECK(sqlite3_column_type(stmt, 0) == SQLITE_BLOB);
    const uint8_t* blob = (const uint8_t*)sqlite3_column_blob(stmt, 0);
    int blob_len = sqlite3_column_bytes(stmt, 0);
    CHECK(blob != NULL);
    CHECK(blob_len > 0);

    const uint8_t* p = blob;
    // LIST, count=2
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_LIST); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;

    // Element 0: TUPLE, count=2
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_TUPLE); p += 8;
    CHECK_EQ(read_u64_le(p), 2); p += 8;
    // I64(7)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_I64); p += 8;
    CHECK_EQ(read_i64_le(p), 7); p += 8;
    // DICT, count=1
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_DICT); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    // Pair: STR("x") -> F64(3.14)
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_STR); p += 8;
    CHECK_EQ(read_u64_le(p), 1); p += 8;
    CHECK(memcmp(p, "x", 1) == 0); p += 8;
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_F64); p += 8;
    CHECK(read_f64_le(p) > 3.13 && read_f64_le(p) < 3.15); p += 8;

    // Element 1: NONE
    CHECK_EQ(read_u64_le(p), TRACE_TYPE_NONE); p += 8;

    CHECK_EQ((int)(p - blob), blob_len);

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

    printf("\nList serialization:\n");
    RUN(list_size_empty);
    RUN(list_size_ints);
    RUN(list_write_ints);
    RUN(list_write_mixed_types);
    RUN(list_write_nested);
    RUN(list_tcp_end_to_end);

    printf("\nTuple serialization:\n");
    RUN(tuple_size_empty);
    RUN(tuple_write_ints);
    RUN(tuple_write_mixed);

    printf("\nDict serialization:\n");
    RUN(dict_size_empty);
    RUN(dict_write_int_values);
    RUN(dict_write_mixed_values);
    RUN(dict_nested_in_list);
    RUN(tuple_and_dict_tcp_end_to_end);

    printf("\nNested stress:\n");
    RUN(nested_stress);
    RUN(nested_stress_repeated_serialize);
    RUN(nested_stress_tcp);

#ifdef TRACER_SQLITE_ENABLED
    printf("\nSQLite subscriber:\n");
    RUN(sqlite_subscriber_writes_events);
    RUN(sqlite_subscriber_correct_values);
    RUN(sqlite_list_tuple_dict);
    RUN(sqlite_nested_containers);
#else
    printf("\n(SQLite tests skipped - TRACER_SQLITE_ENABLED not defined)\n");
#endif

    return test_print_results("Tracer Tests");
}

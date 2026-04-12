#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef _WIN32
#include <signal.h>
#endif

#include "tracing.h"

// ============================================================
// Minimal test framework
// ============================================================

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_current_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    g_current_failed = 0; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    test_##name(); \
    g_tests_run++; \
    if (g_current_failed) { g_tests_failed++; } \
    else { g_tests_passed++; printf("PASS\n"); } \
} while (0)

#define CHECK(cond) do { \
    if (!(cond)) { \
        if (!g_current_failed) printf("FAIL\n"); \
        fprintf(stderr, "    %s:%d: CHECK( %s )\n", __FILE__, __LINE__, #cond); \
        g_current_failed = 1; \
        return; \
    } \
} while (0)

#define CHECK_EQ(a, b) do { \
    long long _a = (long long)(a), _b = (long long)(b); \
    if (_a != _b) { \
        if (!g_current_failed) printf("FAIL\n"); \
        fprintf(stderr, "    %s:%d: CHECK_EQ( %s, %s )  =>  %lld != %lld\n", \
                __FILE__, __LINE__, #a, #b, _a, _b); \
        g_current_failed = 1; \
        return; \
    } \
} while (0)

// ============================================================
// Helpers
// ============================================================

static size_t entry_total_bytes(size_t payload_size) {
    return align_up(payload_size, MPSC_ALLOCATION_ALIGNMENT) + MPSC_HEADER_TOTAL_SIZE;
}

static uint32_t max_entries_for_size(size_t buf_size, size_t payload_size) {
    return (uint32_t)(buf_size / entry_total_bytes(payload_size));
}

static void fill_pattern(uint8_t* buf, size_t size, uint8_t seed) {
    for (size_t i = 0; i < size; i++)
        buf[i] = (uint8_t)(seed + i);
}

static bool verify_pattern(const uint8_t* buf, size_t size, uint8_t seed) {
    for (size_t i = 0; i < size; i++)
        if (buf[i] != (uint8_t)(seed + i)) return false;
    return true;
}

// ============================================================
// Single-threaded tests
// ============================================================

TEST(create_destroy) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    CHECK(rb.ring_buffer != NULL);
    CHECK_EQ(rb.size, 4096);
    CHECK_EQ(atomic_load(&rb.produced_offset), 0);
    CHECK_EQ(atomic_load(&rb.consumed_offset), 0);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(reject_zero_size) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    CHECK(mpsc_ring_buffer_try_reserve_write(&rb, 0) == NULL);
    CHECK_EQ(atomic_load(&rb.produced_offset), 0);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(reject_oversized) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    size_t max_payload = rb.size - MPSC_HEADER_TOTAL_SIZE;
    CHECK(mpsc_ring_buffer_try_reserve_write(&rb, max_payload + 1) == NULL);
    CHECK(mpsc_ring_buffer_try_reserve_write(&rb, rb.size) == NULL);

    // Max valid payload should succeed
    uint8_t* p = mpsc_ring_buffer_try_reserve_write(&rb, max_payload);
    CHECK(p != NULL);
    mpsc_ring_buffer_commit_write(&rb, p);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(single_entry_roundtrip) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    const size_t payload = 64;
    uint8_t* w = mpsc_ring_buffer_try_reserve_write(&rb, payload);
    CHECK(w != NULL);

    fill_pattern(w, payload, 0xAB);
    mpsc_ring_buffer_commit_write(&rb, w);

    uint8_t* data;
    size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    CHECK_EQ(sz, payload);
    CHECK(verify_pattern(data, sz, 0xAB));
    mpsc_ring_buffer_lock_release_read(&rb, sz);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(multiple_sequential_entries) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    for (int i = 0; i < 50; i++) {
        const size_t payload = 32;
        uint8_t* w = mpsc_ring_buffer_try_reserve_write(&rb, payload);
        CHECK(w != NULL);
        fill_pattern(w, payload, (uint8_t)i);
        mpsc_ring_buffer_commit_write(&rb, w);

        uint8_t* data;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
        CHECK_EQ(sz, payload);
        CHECK(verify_pattern(data, sz, (uint8_t)i));
        mpsc_ring_buffer_lock_release_read(&rb, sz);
    }

    mpsc_ring_buffer_destroy(&rb);
}

TEST(fill_buffer_returns_null) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    const size_t payload = 8;
    uint32_t max_n = max_entries_for_size(rb.size, payload);
    CHECK(max_n > 0 && max_n <= 256);

    uint8_t* ptrs[256];
    for (uint32_t i = 0; i < max_n; i++) {
        ptrs[i] = mpsc_ring_buffer_try_reserve_write(&rb, payload);
        CHECK(ptrs[i] != NULL);
        mpsc_ring_buffer_commit_write(&rb, ptrs[i]);
    }

    // Buffer full
    CHECK(mpsc_ring_buffer_try_reserve_write(&rb, payload) == NULL);
    CHECK(mpsc_ring_buffer_try_reserve_write(&rb, 1) == NULL);

    // Consume one to free space
    uint8_t* data;
    size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    mpsc_ring_buffer_lock_release_read(&rb, sz);

    // Should be able to allocate again
    uint8_t* w = mpsc_ring_buffer_try_reserve_write(&rb, payload);
    CHECK(w != NULL);
    mpsc_ring_buffer_commit_write(&rb, w);

    // Drain remaining
    for (uint32_t i = 0; i < max_n; i++) {
        sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
        mpsc_ring_buffer_lock_release_read(&rb, sz);
    }

    mpsc_ring_buffer_destroy(&rb);
}

TEST(ring_wrap_around) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    const size_t payload = 64;
    // 4096 / entry_total_bytes(64) = 4096 / 80 = 51 per buffer
    // 200 entries wraps ~4 times
    for (int i = 0; i < 200; i++) {
        uint8_t* w = mpsc_ring_buffer_try_reserve_write(&rb, payload);
        CHECK(w != NULL);
        fill_pattern(w, payload, (uint8_t)(i & 0xFF));
        mpsc_ring_buffer_commit_write(&rb, w);

        uint8_t* data;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
        CHECK_EQ(sz, payload);
        CHECK(verify_pattern(data, sz, (uint8_t)(i & 0xFF)));
        mpsc_ring_buffer_lock_release_read(&rb, sz);
    }

    mpsc_ring_buffer_destroy(&rb);
}

TEST(various_payload_sizes) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 8192);

    size_t sizes[] = {1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 48, 63, 64, 100, 128, 255, 256};
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int i = 0; i < n; i++) {
        uint8_t* w = mpsc_ring_buffer_try_reserve_write(&rb, sizes[i]);
        CHECK(w != NULL);
        fill_pattern(w, sizes[i], (uint8_t)(i * 37));
        mpsc_ring_buffer_commit_write(&rb, w);

        uint8_t* data;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
        CHECK_EQ(sz, sizes[i]);
        CHECK(verify_pattern(data, sz, (uint8_t)(i * 37)));
        mpsc_ring_buffer_lock_release_read(&rb, sz);
    }

    mpsc_ring_buffer_destroy(&rb);
}

TEST(batch_reserve_commit_consume) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    const size_t payload = 16;
    const int batch = 10;
    uint8_t* ptrs[10];

    for (int i = 0; i < batch; i++) {
        ptrs[i] = mpsc_ring_buffer_try_reserve_write(&rb, payload);
        CHECK(ptrs[i] != NULL);
        fill_pattern(ptrs[i], payload, (uint8_t)i);
    }

    // Commit all in allocation order
    for (int i = 0; i < batch; i++)
        mpsc_ring_buffer_commit_write(&rb, ptrs[i]);

    // Consume all - must come out in allocation order
    for (int i = 0; i < batch; i++) {
        uint8_t* data;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
        CHECK_EQ(sz, payload);
        CHECK(verify_pattern(data, sz, (uint8_t)i));
        mpsc_ring_buffer_lock_release_read(&rb, sz);
    }

    mpsc_ring_buffer_destroy(&rb);
}

TEST(out_of_order_commit) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    const size_t payload = 16;

    uint8_t* a = mpsc_ring_buffer_try_reserve_write(&rb, payload);
    uint8_t* b = mpsc_ring_buffer_try_reserve_write(&rb, payload);
    uint8_t* c = mpsc_ring_buffer_try_reserve_write(&rb, payload);
    CHECK(a != NULL);
    CHECK(b != NULL);
    CHECK(c != NULL);

    fill_pattern(a, payload, 0xAA);
    fill_pattern(b, payload, 0xBB);
    fill_pattern(c, payload, 0xCC);

    // Commit in reverse order
    mpsc_ring_buffer_commit_write(&rb, c);
    mpsc_ring_buffer_commit_write(&rb, b);
    mpsc_ring_buffer_commit_write(&rb, a);

    // Consume: must get A, B, C (allocation order, not commit order)
    uint8_t* data;
    size_t sz;

    sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    CHECK_EQ(sz, payload);
    CHECK(verify_pattern(data, sz, 0xAA));
    mpsc_ring_buffer_lock_release_read(&rb, sz);

    sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    CHECK_EQ(sz, payload);
    CHECK(verify_pattern(data, sz, 0xBB));
    mpsc_ring_buffer_lock_release_read(&rb, sz);

    sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    CHECK_EQ(sz, payload);
    CHECK(verify_pattern(data, sz, 0xCC));
    mpsc_ring_buffer_lock_release_read(&rb, sz);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(payload_alignment) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    size_t sizes[] = {1, 3, 5, 7, 9, 13, 15, 17, 31, 33};
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int i = 0; i < n; i++) {
        uint8_t* w = mpsc_ring_buffer_try_reserve_write(&rb, sizes[i]);
        CHECK(w != NULL);
        // Returned pointer must be aligned
        CHECK(((uintptr_t)w & (MPSC_ALLOCATION_ALIGNMENT - 1)) == 0);
        mpsc_ring_buffer_commit_write(&rb, w);

        uint8_t* data;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
        CHECK_EQ(sz, sizes[i]);
        mpsc_ring_buffer_lock_release_read(&rb, sz);
    }

    mpsc_ring_buffer_destroy(&rb);
}

TEST(offsets_advance_correctly) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    const size_t payload = 32;
    uint32_t units = (uint32_t)(entry_total_bytes(payload) >> MPSC_ALLOCATION_ALIGNMENT_BITS);

    uint8_t* a = mpsc_ring_buffer_try_reserve_write(&rb, payload);
    CHECK_EQ(atomic_load(&rb.produced_offset), units);

    uint8_t* b = mpsc_ring_buffer_try_reserve_write(&rb, payload);
    CHECK_EQ(atomic_load(&rb.produced_offset), 2 * units);

    // consumed still at 0
    CHECK_EQ(atomic_load(&rb.consumed_offset), 0);

    mpsc_ring_buffer_commit_write(&rb, a);
    mpsc_ring_buffer_commit_write(&rb, b);

    uint8_t* data;
    size_t sz;

    sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    mpsc_ring_buffer_lock_release_read(&rb, sz);
    CHECK_EQ(atomic_load(&rb.consumed_offset), units);

    sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    mpsc_ring_buffer_lock_release_read(&rb, sz);
    CHECK_EQ(atomic_load(&rb.consumed_offset), 2 * units);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(wait_reserve_basic) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    uint8_t* w = mpsc_ring_buffer_wait_reserve_write(&rb, 64);
    CHECK(w != NULL);
    fill_pattern(w, 64, 0x42);
    mpsc_ring_buffer_commit_write(&rb, w);

    uint8_t* data;
    size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
    CHECK_EQ(sz, 64);
    CHECK(verify_pattern(data, 64, 0x42));
    mpsc_ring_buffer_lock_release_read(&rb, sz);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(wait_reserve_rejects_invalid) {
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    CHECK(mpsc_ring_buffer_wait_reserve_write(&rb, 0) == NULL);
    CHECK(mpsc_ring_buffer_wait_reserve_write(&rb, rb.size) == NULL);

    mpsc_ring_buffer_destroy(&rb);
}

TEST(consume_after_wrap_data_integrity) {
    // Stress the wrap boundary with careful data checks
    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, 4096);

    const size_t payload = 100; // not a power of 2, entry = 112+16 = 128 bytes
    // 4096 / 128 = 32 entries per buffer

    for (int round = 0; round < 5; round++) {
        // Fill buffer completely
        uint32_t max_n = max_entries_for_size(rb.size, payload);
        uint8_t* ptrs[64];
        CHECK(max_n <= 64);

        for (uint32_t i = 0; i < max_n; i++) {
            ptrs[i] = mpsc_ring_buffer_try_reserve_write(&rb, payload);
            CHECK(ptrs[i] != NULL);
            fill_pattern(ptrs[i], payload, (uint8_t)(round * 50 + i));
            mpsc_ring_buffer_commit_write(&rb, ptrs[i]);
        }

        // Drain all
        for (uint32_t i = 0; i < max_n; i++) {
            uint8_t* data;
            size_t sz = mpsc_ring_buffer_lock_acquire_read(&rb, &data);
            CHECK_EQ(sz, payload);
            CHECK(verify_pattern(data, sz, (uint8_t)(round * 50 + i)));
            mpsc_ring_buffer_lock_release_read(&rb, sz);
        }
    }

    mpsc_ring_buffer_destroy(&rb);
}

// ============================================================
// Multi-threaded tests
// ============================================================

typedef struct {
    uint32_t producer_id;
    uint32_t sequence;
} TestPayload;

typedef struct {
    MpscRingBuffer* rb;
    uint32_t producer_id;
    uint32_t count;
} ProducerArgs;

typedef struct {
    MpscRingBuffer* rb;
    uint32_t expected_total;
    uint32_t num_producers;
    atomic_int error_count;
    atomic_uint next_seq[64]; // expected next sequence per producer
} ConsumerResult;

THREAD_PROC(producer_fixed_thread) {
    ProducerArgs* args = (ProducerArgs*)data;

    for (uint32_t i = 0; i < args->count; i++) {
        uint8_t* w = mpsc_ring_buffer_wait_reserve_write(args->rb, sizeof(TestPayload));
        TestPayload* p = (TestPayload*)w;
        p->producer_id = args->producer_id;
        p->sequence    = i;
        mpsc_ring_buffer_commit_write(args->rb, w);
    }

    return 0;
}

THREAD_PROC(consumer_fixed_thread) {
    ConsumerResult* res = (ConsumerResult*)data;

    for (uint32_t i = 0; i < res->expected_total; i++) {
        uint8_t* rdata;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(res->rb, &rdata);

        if (sz != sizeof(TestPayload)) {
            atomic_fetch_add(&res->error_count, 1);
            mpsc_ring_buffer_lock_release_read(res->rb, sz);
            continue;
        }

        TestPayload* p = (TestPayload*)rdata;
        uint32_t pid = p->producer_id;
        uint32_t seq = p->sequence;

        if (pid >= res->num_producers) {
            atomic_fetch_add(&res->error_count, 1);
        } else {
            uint32_t expected = atomic_load(&res->next_seq[pid]);
            if (seq != expected) {
                atomic_fetch_add(&res->error_count, 1);
            }
            atomic_store(&res->next_seq[pid], seq + 1);
        }

        mpsc_ring_buffer_lock_release_read(res->rb, sz);
    }

    return 0;
}

static void run_mpsc_test(uint32_t num_producers, uint32_t entries_per_producer, size_t buf_size) {
    ASSERT(num_producers <= 64);

    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, buf_size);

    uint32_t total = num_producers * entries_per_producer;

    ConsumerResult cres;
    memset(&cres, 0, sizeof(cres));
    cres.rb             = &rb;
    cres.expected_total = total;
    cres.num_producers  = num_producers;
    atomic_store(&cres.error_count, 0);
    for (uint32_t i = 0; i < num_producers; i++)
        atomic_store(&cres.next_seq[i], 0);

    Thread consumer;
    create_thread(consumer_fixed_thread, &cres, &consumer);

    ProducerArgs pargs[64];
    Thread producers[64];
    for (uint32_t i = 0; i < num_producers; i++) {
        pargs[i].rb          = &rb;
        pargs[i].producer_id = i;
        pargs[i].count       = entries_per_producer;
        create_thread(producer_fixed_thread, &pargs[i], &producers[i]);
    }

    for (uint32_t i = 0; i < num_producers; i++)
        join_thread(&producers[i]);
    join_thread(&consumer);

    // Verify after all threads joined
    int errors = atomic_load(&cres.error_count);
    mpsc_ring_buffer_destroy(&rb);

    CHECK_EQ(errors, 0);
    for (uint32_t i = 0; i < num_producers; i++)
        CHECK_EQ(atomic_load(&cres.next_seq[i]), entries_per_producer);
}

TEST(mt_spsc_basic) {
    run_mpsc_test(1, 1000, 4096);
}

TEST(mt_spsc_large_buffer) {
    run_mpsc_test(1, 10000, 1 << 16);
}

TEST(mt_two_producers) {
    run_mpsc_test(2, 1000, 4096);
}

TEST(mt_four_producers) {
    run_mpsc_test(4, 1000, 4096);
}

TEST(mt_many_producers) {
    run_mpsc_test(16, 500, 4096);
}

TEST(mt_large_buffer_many_producers) {
    run_mpsc_test(8, 5000, 1 << 16);
}

TEST(mt_producers_wait_for_space) {
    // Tiny buffer: 4 producers * 500 entries >> 128 capacity
    // Forces producers to block until consumer frees space
    run_mpsc_test(4, 500, 4096);
}

// --- Variable-size stress test ---

typedef struct {
    MpscRingBuffer* rb;
    uint32_t producer_id;
    uint32_t count;
} StressProducerArgs;

typedef struct {
    MpscRingBuffer* rb;
    uint32_t expected_total;
    uint32_t num_producers;
    atomic_int error_count;
    atomic_uint next_seq[64];
} StressConsumerResult;

THREAD_PROC(stress_producer_thread) {
    StressProducerArgs* args = (StressProducerArgs*)data;

    for (uint32_t i = 0; i < args->count; i++) {
        // Vary payload between 17 and 216 bytes
        size_t payload_size = 17 + ((args->producer_id * 7 + i * 13) % 200);

        uint8_t* w = mpsc_ring_buffer_wait_reserve_write(args->rb, payload_size);

        // Header: producer_id + sequence
        ((uint32_t*)w)[0] = args->producer_id;
        ((uint32_t*)w)[1] = i;

        // Fill remaining bytes with verifiable pattern
        uint8_t seed = (uint8_t)(args->producer_id ^ i);
        for (size_t j = 8; j < payload_size; j++)
            w[j] = (uint8_t)(seed + j);

        mpsc_ring_buffer_commit_write(args->rb, w);
    }

    return 0;
}

THREAD_PROC(stress_consumer_thread) {
    StressConsumerResult* res = (StressConsumerResult*)data;

    for (uint32_t i = 0; i < res->expected_total; i++) {
        uint8_t* rdata;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(res->rb, &rdata);

        if (sz < 8) {
            atomic_fetch_add(&res->error_count, 1);
            mpsc_ring_buffer_lock_release_read(res->rb, sz);
            continue;
        }

        uint32_t pid = ((uint32_t*)rdata)[0];
        uint32_t seq = ((uint32_t*)rdata)[1];

        if (pid >= res->num_producers) {
            atomic_fetch_add(&res->error_count, 1);
        } else {
            // Check per-producer ordering
            uint32_t expected = atomic_load(&res->next_seq[pid]);
            if (seq != expected)
                atomic_fetch_add(&res->error_count, 1);
            atomic_store(&res->next_seq[pid], seq + 1);

            // Verify payload pattern
            uint8_t seed = (uint8_t)(pid ^ seq);
            for (size_t j = 8; j < sz; j++) {
                if (rdata[j] != (uint8_t)(seed + j)) {
                    atomic_fetch_add(&res->error_count, 1);
                    break;
                }
            }
        }

        mpsc_ring_buffer_lock_release_read(res->rb, sz);
    }

    return 0;
}

TEST(mt_stress_variable_sizes) {
    uint32_t num_producers = 8;
    uint32_t entries_per   = 2000;
    size_t buf_size        = 1 << 16; // 64 KiB

    MpscRingBuffer rb;
    mpsc_ring_buffer_create(&rb, buf_size);

    uint32_t total = num_producers * entries_per;

    StressConsumerResult cres;
    memset(&cres, 0, sizeof(cres));
    cres.rb             = &rb;
    cres.expected_total = total;
    cres.num_producers  = num_producers;
    atomic_store(&cres.error_count, 0);
    for (uint32_t i = 0; i < num_producers; i++)
        atomic_store(&cres.next_seq[i], 0);

    Thread consumer;
    create_thread(stress_consumer_thread, &cres, &consumer);

    StressProducerArgs pargs[64];
    Thread producers[64];
    for (uint32_t i = 0; i < num_producers; i++) {
        pargs[i].rb          = &rb;
        pargs[i].producer_id = i;
        pargs[i].count       = entries_per;
        create_thread(stress_producer_thread, &pargs[i], &producers[i]);
    }

    for (uint32_t i = 0; i < num_producers; i++)
        join_thread(&producers[i]);
    join_thread(&consumer);

    int errors = atomic_load(&cres.error_count);
    mpsc_ring_buffer_destroy(&rb);

    CHECK_EQ(errors, 0);
    for (uint32_t i = 0; i < num_producers; i++)
        CHECK_EQ(atomic_load(&cres.next_seq[i]), entries_per);
}

// ============================================================
// main
// ============================================================

#ifndef _WIN32
static void timeout_handler(int sig) {
    (void)sig;
    fprintf(stderr, "\nTEST TIMEOUT -- possible deadlock\n");
    _exit(2);
}
#endif

int main(void) {
#ifndef _WIN32
    signal(SIGALRM, timeout_handler);
    alarm(60);
#endif

    printf("=== MPSC Ring Buffer Tests ===\n\n");

    printf("Single-threaded:\n");
    RUN(create_destroy);
    RUN(reject_zero_size);
    RUN(reject_oversized);
    RUN(single_entry_roundtrip);
    RUN(multiple_sequential_entries);
    RUN(fill_buffer_returns_null);
    RUN(ring_wrap_around);
    RUN(various_payload_sizes);
    RUN(batch_reserve_commit_consume);
    RUN(out_of_order_commit);
    RUN(payload_alignment);
    RUN(offsets_advance_correctly);
    RUN(wait_reserve_basic);
    RUN(wait_reserve_rejects_invalid);
    RUN(consume_after_wrap_data_integrity);

    printf("\nMulti-threaded:\n");
    RUN(mt_spsc_basic);
    RUN(mt_spsc_large_buffer);
    RUN(mt_two_producers);
    RUN(mt_four_producers);
    RUN(mt_many_producers);
    RUN(mt_large_buffer_many_producers);
    RUN(mt_producers_wait_for_space);
    RUN(mt_stress_variable_sizes);

    printf("\n=== %d/%d passed", g_tests_passed, g_tests_run);
    if (g_tests_failed > 0)
        printf(", %d FAILED", g_tests_failed);
    printf(" ===\n");

    return g_tests_failed > 0 ? 1 : 0;
}

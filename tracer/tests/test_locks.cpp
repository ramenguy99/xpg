#define TRACER_IMPLEMENTATION
#include "tracer.h"

extern "C" {
#include "testing.h"
}

// ============================================================
// Futex tests
// ============================================================

TEST(futex_init_load) {
    Futex f;
    futex_init(&f, 42);
    CHECK_EQ(atomic_load_explicit(&f, memory_order_relaxed), 42);
}

TEST(futex_wake_without_waiter) {
    Futex f;
    futex_init(&f, 0);
    // Should not hang or crash
    futex_wake(&f);
    futex_wake_all(&f);
}

struct FutexThreadData {
    Futex* futex;
    _Atomic(int) started;
    _Atomic(int) done;
};

static THREAD_PROC(futex_waiter_thread) {
    FutexThreadData* d = (FutexThreadData*)data;
    atomic_store_explicit(&d->started, 1, memory_order_release);
    futex_wait(d->futex, 0);
    atomic_store_explicit(&d->done, 1, memory_order_release);
    return 0;
}

TEST(futex_wait_wake) {
    Futex f;
    futex_init(&f, 0);

    FutexThreadData td;
    td.futex = &f;
    atomic_store_explicit(&td.started, 0, memory_order_relaxed);
    atomic_store_explicit(&td.done, 0, memory_order_relaxed);

    Thread t;
    create_thread(futex_waiter_thread, &td, &t);

    // Wait for thread to start
    while (!atomic_load_explicit(&td.started, memory_order_acquire)) {
        thread_sleep_ms(1);
    }
    thread_sleep_ms(10);

    // Thread should be blocked
    CHECK_EQ(atomic_load_explicit(&td.done, memory_order_acquire), 0);

    // Change value and wake
    atomic_store_explicit(&f, 1, memory_order_release);
    futex_wake(&f);

    join_thread(&t);
    CHECK_EQ(atomic_load_explicit(&td.done, memory_order_acquire), 1);
}

TEST(futex_spurious_no_block) {
    Futex f;
    futex_init(&f, 5);
    // Wait with non-matching expected should return immediately
    bool ret = futex_wait(&f, 99);
    (void)ret;
    // If we get here, we didn't block forever
    CHECK(true);
}

// ============================================================
// Mutex tests
// ============================================================

TEST(mutex_lock_unlock) {
    Mutex m;
    mutex_init(&m);
    mutex_lock(&m);
    mutex_unlock(&m);
    CHECK(true);
}

TEST(mutex_try_lock) {
    Mutex m;
    mutex_init(&m);

    CHECK(mutex_try_lock(&m));
    // Already locked, try_lock should fail
    CHECK(!mutex_try_lock(&m));
    mutex_unlock(&m);

    // Now it should succeed again
    CHECK(mutex_try_lock(&m));
    mutex_unlock(&m);
}

struct MutexCounterData {
    Mutex* mutex;
    int* counter;
    int iterations;
};

static THREAD_PROC(mutex_counter_thread) {
    MutexCounterData* d = (MutexCounterData*)data;
    for (int i = 0; i < d->iterations; i++) {
        mutex_lock(d->mutex);
        (*d->counter)++;
        mutex_unlock(d->mutex);
    }
    return 0;
}

TEST(mutex_contended) {
    Mutex m;
    mutex_init(&m);
    int counter = 0;

    const int num_threads = 4;
    const int iters = 10000;

    MutexCounterData args[4];
    Thread threads[4];

    for (int i = 0; i < num_threads; i++) {
        args[i].mutex = &m;
        args[i].counter = &counter;
        args[i].iterations = iters;
        create_thread(mutex_counter_thread, &args[i], &threads[i]);
    }

    for (int i = 0; i < num_threads; i++)
        join_thread(&threads[i]);

    CHECK_EQ(counter, num_threads * iters);
}

// ============================================================
// RWLock tests
// ============================================================

TEST(rwlock_read_write_basic) {
    RWLock rw;
    rwlock_init(&rw);

    rwlock_read(&rw);
    rwlock_read_unlock(&rw);

    rwlock_write(&rw);
    rwlock_write_unlock(&rw);

    CHECK(true);
}

TEST(rwlock_multiple_readers) {
    RWLock rw;
    rwlock_init(&rw);

    // Multiple concurrent read locks
    rwlock_read(&rw);
    rwlock_read(&rw);
    rwlock_read(&rw);
    rwlock_read_unlock(&rw);
    rwlock_read_unlock(&rw);
    rwlock_read_unlock(&rw);

    CHECK(true);
}

TEST(rwlock_try_read_while_write_locked) {
    RWLock rw;
    rwlock_init(&rw);

    rwlock_write(&rw);
    CHECK(!rwlock_try_read(&rw));
    rwlock_write_unlock(&rw);

    CHECK(rwlock_try_read(&rw));
    rwlock_read_unlock(&rw);
}

TEST(rwlock_try_write_while_read_locked) {
    RWLock rw;
    rwlock_init(&rw);

    rwlock_read(&rw);
    CHECK(!rwlock_try_write(&rw));
    rwlock_read_unlock(&rw);

    CHECK(rwlock_try_write(&rw));
    rwlock_write_unlock(&rw);
}

struct RWLockData {
    RWLock* rw;
    _Atomic(int)* shared_value;
    _Atomic(int)* error_count;
    int iterations;
};

static THREAD_PROC(rwlock_writer_thread) {
    RWLockData* d = (RWLockData*)data;
    for (int i = 0; i < d->iterations; i++) {
        rwlock_write(d->rw);
        atomic_store_explicit(d->shared_value, i, memory_order_relaxed);
        thread_sleep_ms(0);
        int v = atomic_load_explicit(d->shared_value, memory_order_relaxed);
        if (v != i) atomic_fetch_add_explicit(d->error_count, 1, memory_order_relaxed);
        rwlock_write_unlock(d->rw);
    }
    return 0;
}

static THREAD_PROC(rwlock_reader_thread) {
    RWLockData* d = (RWLockData*)data;
    for (int i = 0; i < d->iterations; i++) {
        rwlock_read(d->rw);
        int v1 = atomic_load_explicit(d->shared_value, memory_order_relaxed);
        thread_sleep_ms(0);
        int v2 = atomic_load_explicit(d->shared_value, memory_order_relaxed);
        if (v1 != v2) atomic_fetch_add_explicit(d->error_count, 1, memory_order_relaxed);
        rwlock_read_unlock(d->rw);
    }
    return 0;
}

TEST(rwlock_concurrent_readers_writers) {
    RWLock rw;
    rwlock_init(&rw);
    _Atomic(int) shared_value;
    _Atomic(int) error_count;
    atomic_store_explicit(&shared_value, 0, memory_order_relaxed);
    atomic_store_explicit(&error_count, 0, memory_order_relaxed);

    const int num_readers = 4;
    const int num_writers = 2;
    const int iters = 500;

    RWLockData args[6];
    Thread threads[6];

    for (int i = 0; i < num_writers; i++) {
        args[i].rw = &rw;
        args[i].shared_value = &shared_value;
        args[i].error_count = &error_count;
        args[i].iterations = iters;
        create_thread(rwlock_writer_thread, &args[i], &threads[i]);
    }
    for (int i = 0; i < num_readers; i++) {
        int idx = num_writers + i;
        args[idx].rw = &rw;
        args[idx].shared_value = &shared_value;
        args[idx].error_count = &error_count;
        args[idx].iterations = iters;
        create_thread(rwlock_reader_thread, &args[idx], &threads[idx]);
    }

    for (int i = 0; i < num_writers + num_readers; i++)
        join_thread(&threads[i]);

    CHECK_EQ(atomic_load_explicit(&error_count, memory_order_relaxed), 0);
}

TEST(rwlock_downgrade) {
    RWLock rw;
    rwlock_init(&rw);

    rwlock_write(&rw);
    downgrade(&rw);
    // Now it's read-locked, another reader should be able to lock
    CHECK(rwlock_try_read(&rw));
    rwlock_read_unlock(&rw);
    rwlock_read_unlock(&rw);
}

// ============================================================
// main
// ============================================================

int main() {
    test_setup_timeout(30);

    printf("=== Lock Tests ===\n\n");

    printf("Futex:\n");
    RUN(futex_init_load);
    RUN(futex_wake_without_waiter);
    RUN(futex_wait_wake);
    RUN(futex_spurious_no_block);

    printf("\nMutex:\n");
    RUN(mutex_lock_unlock);
    RUN(mutex_try_lock);
    RUN(mutex_contended);

    printf("\nRWLock:\n");
    RUN(rwlock_read_write_basic);
    RUN(rwlock_multiple_readers);
    RUN(rwlock_try_read_while_write_locked);
    RUN(rwlock_try_write_while_read_locked);
    RUN(rwlock_concurrent_readers_writers);
    RUN(rwlock_downgrade);

    return test_print_results("Lock Tests");
}

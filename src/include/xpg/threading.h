#pragma once

#include <atomic>
#include <atomic_queue/atomic_queue.h>

#include "platform.h"

#ifdef __aarch64__
#ifdef __clang__
#include <arm_acle.h>
#define SpinlockHint() __yield()
#else
#define SpinlockHint() do {} while(0)
#endif
#else
#define SpinlockHint() _mm_pause()
#endif

// struct ConditionVariable {
// #ifdef _WIN32
//     CRITICAL_SECTION lock;
//     CONDITION_VARIABLE cv;
// #else
//     int fd;
// #endif // _WIN32
//
//     ConditionVariable() {
// #ifdef _WIN32
//         InitializeCriticalSection(&lock);
//         InitializeConditionVariable(&cv);
// #endif
//     }
//
//     void signal() {
//     }
//
//     void wait() {
//     }
// };

#ifdef _WIN32
#define THREAD_PROC(x) DWORD WINAPI x(LPVOID data)
#else
#define THREAD_PROC(x) void* x(void* data)
#endif // _WIN32

namespace xpg {

typedef THREAD_PROC(ThreadProc);

struct Semaphore {
#ifdef _WIN32
    HANDLE handle;

    Semaphore(): handle(INVALID_HANDLE_VALUE) { }

    void init(u32 initial_count, u32 max_count) {
        assert(handle == INVALID_HANDLE_VALUE);
        handle = CreateSemaphoreEx(0, initial_count, max_count, 0, 0, SEMAPHORE_ALL_ACCESS);
    }

    void destroy() {
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
    }

    ~Semaphore() {
        destroy();
    }

    void acquire() {
       WaitForSingleObjectEx(handle, INFINITE, FALSE);
    }

    void release() {
        ReleaseSemaphore(handle, 1, 0);
    }

#else
    sem_t semaphore;
    bool initialized;

    Semaphore(): initialized(false) { }

    void init(u32 initial_count, u32 max_count) {
        assert(!initialized);
        sem_init(&semaphore, 0, initial_count);
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            sem_close(&semaphore);
            initialized = false;
        }
    }

    void acquire() {
        sem_wait(&semaphore);
    }

    void release() {
        sem_post(&semaphore);
    }
#endif
};

struct alignas(64) BlockingCounter {
    std::atomic<usize> count;
    Semaphore semaphore;

    BlockingCounter() {
        semaphore.init(0, 1);
    }

    void arm(usize count) {
        this->count.store(count, std::memory_order_relaxed);
    }

    void signal() {
        usize prev_count = count.fetch_sub(1, std::memory_order_relaxed);
        if (prev_count == 1) {
            semaphore.release();
        }
    }

    void wait() {
        semaphore.acquire();
    }
};

struct Thread {
#ifdef _WIN32
    HANDLE handle;

    Thread(ThreadProc proc, void* data) {
        handle = CreateThread(0, 0, proc, data, 0, 0);
    }

    void join() {
        if (handle != INVALID_HANDLE_VALUE) {
            WaitForSingleObject(handle, INFINITE);
            CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
    }
#else
    pthread_t thread;
    Thread(ThreadProc proc, void* data) {
        pthread_create(&thread, 0, proc, data);
    }

    void join() {
        pthread_join(thread, 0);
    }
#endif
};

struct WorkerPool {
    // Safety margin to avoid forkbombing the system.
    // Real max value would be UINT32_MAX.
    static constexpr u32 MAX_THREADS = 1024 * 1024;

    struct WorkerInfo {
        WorkerPool* pool;
        usize index;
        void* data;
    };

    struct Work {
        void (*callback)(WorkerInfo* worker_info, void* user_data);
        void* user_data;
    };

    atomic_queue::AtomicQueue2<Work, 1024> queue;
    Array<Thread> workers;
    Array<WorkerInfo> worker_info;

    Semaphore semaphore; // Semaphore for waiting for work.
    std::atomic<bool> exit;

    static THREAD_PROC(thread_proc) {
        WorkerInfo* info = (WorkerInfo*)data;
        WorkerPool* pool = info->pool;
        while (true) {
            if (pool->exit.load()) {
                break;
            }

            Work w;
            while (pool->queue.try_pop(w)) {
                if (pool->exit.load()) {
                    goto done;
                }

                // Do work with item
                w.callback(info, w.user_data);
            }

            // Wait to be waked up again.
            pool->semaphore.acquire();
        }
        done:

        return 0;
    }

    WorkerPool() : exit(false) {}
    ~WorkerPool() {
        destroy();
    }

    void init(u32 num_workers)
    {
        assert(num_workers <= MAX_THREADS);
        semaphore.init(0, num_workers);

        // Initialize workers
        worker_info.resize(num_workers);
        workers.resize(num_workers);
        for (usize i = 0; i < workers.length; i++) {
            worker_info[i] = {
                .pool = this,
                .index = i,
                .data = nullptr,
            };
            Thread t(thread_proc, &worker_info[i]);
            workers[i] = t;
        }
    }

    void init_with_worker_data(ArrayView<void*> worker_data)
    {
        assert(worker_data.length <= MAX_THREADS);
        semaphore.init(0, (u32)worker_data.length);

        // Initialize workers
        worker_info.resize(worker_data.length);
        workers.resize(worker_data.length);
        for (usize i = 0; i < worker_data.length; i++) {
            worker_info[i] = {
                .pool = this,
                .index = i,
                .data = worker_data[i],
            };
            Thread t(thread_proc, &worker_info[i]);
            workers[i] = t;
        }
    }

    void add_work(Work w) {
        assert(workers.length > 0);

        queue.push(w);
        semaphore.release();
    }

    void destroy() {
        // If already destroyed exit
        if (exit.load(std::memory_order_relaxed)) {
            return;
        }

        // Send quit message to all threads.
        exit.store(true, std::memory_order_relaxed);

        // Signal all threads to exit
        for (usize i = 0; i < workers.length; i++) {
            semaphore.release();
        }

        // Wait for all threads to exit
        for (usize i = 0; i < workers.length; i++) {
            workers[i].join();
        }
    }
};

} // namespace xpg

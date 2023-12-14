#define SpinlockHint() _mm_pause()

struct ConditionVariable {
#ifdef _WIN32
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE cv;
#else
    int fd;
#endif // _WIN32

    ConditionVariable() {
#ifdef _WIN32
        InitializeCriticalSection(&lock);
        InitializeConditionVariable(&cv);
#endif
    }

    void signal() {
    }

    void wait() {
    }
};

#ifdef _WIN32
#define THREAD_PROC(x) DWORD WINAPI x(LPVOID data)
#else
#define THREAD_PROC(x) void* x(void* data)
#endif // _WIN32

typedef THREAD_PROC(ThreadProc);

struct Semaphore {
#ifdef _WIN32
    HANDLE handle;

    Semaphore(u32 initial_count, u32 max_count) {
        handle = CreateSemaphoreEx(0, initial_count, max_count, 0, 0, SEMAPHORE_ALL_ACCESS);
    }

    void acquire() {
       WaitForSingleObjectEx(handle, INFINITE, FALSE);
    }

    void release() {
        ReleaseSemaphore(handle, 1, 0);
    }
#else
    sem_t semaphore;

    Semaphore(u32 initial_count, u32 max_count) {
        sem_init(&semaphore, 0, initial_count);
    }

    void acquire() {
        sem_wait(&semaphore);
    }

    void release() {
        sem_post(&semaphore);
    }
#endif
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
    struct Work {
        void (*callback)(void* user_data);
        void* user_data;
    };

    atomic_queue::AtomicQueue2<Work, 1024> queue;
    Array<Thread> workers;
    Semaphore semaphore; // Semaphore for waiting for work.
    std::atomic<bool> exit;

    static THREAD_PROC(thread_proc) {
        WorkerPool* pool = (WorkerPool*)data;
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
                w.callback(w.user_data);
            }

            // Wait to be waked up again.
            pool->semaphore.acquire();
        }
        done:

        return 0;
    }

    WorkerPool(u32 num_workers):
        semaphore(0, num_workers),
        exit(false)
    {
        // Initialize workers
        workers.resize(num_workers);
        for (usize i = 0; i < workers.length; i++) {
            Thread t(thread_proc, this);
            workers[i] = t;
        }
    }

    void add_work(Work w) {
        queue.push(w);
        semaphore.release();
    }

    void destroy() {
        // Send quit message to all threads.
        exit.store(true, std::memory_order_seq_cst);

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


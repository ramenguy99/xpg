#ifndef TRACING_H
#define TRACING_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>

#define poll WSAPoll
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif

#else

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <poll.h>
#include <sys/uio.h>

#include <pthread.h>

#endif

// Futex
#ifdef __APPLE__
#include <os/os_sync_wait_on_address.h>
#endif


// Ring buffer
#if defined(_WIN32)
#	include <windows.h>
#	pragma comment (lib, "onecore")
#elif defined(__APPLE__)
#	include <mach/mach.h>
#	include <mach/mach_vm.h>
#	include <mach/vm_map.h>
#	include <mach/vm_page_size.h>
#else
#	include <sys/syscall.h>
#	include <sys/mman.h>
#	include <unistd.h>
#	include <fcntl.h>
#endif

// Assert
#if !defined(ASSERT)
#	if defined(_MSC_VER)
#		if !defined(NDEBUG)
#			include <intrin.h>
#			define ASSERT(cond) do { if (!(cond)) __debugbreak(); } while (0)
#		else
#			define ASSERT(cond) do { (void)sizeof(cond); } while (0)
#		endif
#	else
#		include <assert.h>
#		define ASSERT(cond) assert(cond)
#	endif
#endif

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

#ifdef __cplusplus
#include <atomic>
#define _Atomic(T) std::atomic<T>
#define atomic_uint std::atomic<unsigned int>
#define atomic_load_explicit(p, o) (p)->load(o)
#define atomic_store_explicit(p, v, o) (p)->store(v, o)
#define atomic_exchange_explicit(p, v, o) (p)->exchange(v, o)
#define atomic_compare_exchange_strong_explicit(p, e, d, s, f) (p)->compare_exchange_strong(*(e), d, s, f)
#define atomic_compare_exchange_weak_explicit(p, e, d, s, f) (p)->compare_exchange_weak(*(e), d, s, f)
#define atomic_fetch_add_explicit(p, v, o) (p)->fetch_add(v, o)
#define atomic_fetch_sub_explicit(p, v, o) (p)->fetch_sub(v, o)
#define atomic_fetch_or_explicit(p, v, o)  (p)->fetch_or(v, o)
#define atomic_fetch_and_explicit(p, v, o) (p)->fetch_and(v, o)
using std::memory_order_relaxed;
using std::memory_order_acquire;
using std::memory_order_release;
#else
#include <stdatomic.h>
#endif

// Socket utils
#ifdef _WIN32
typedef SOCKET socket_t;
#else
typedef int socket_t;
#endif

#define CACHE_LINE_SIZE 64

// Tracer compile-time configuration
#ifndef TRACER_MAX_SUBSCRIBERS
#define TRACER_MAX_SUBSCRIBERS 4
#endif
#ifndef TRACER_MAX_TRACEPOINTS
#define TRACER_MAX_TRACEPOINTS 256
#endif
#ifndef TRACER_HASH_TABLE_CAPACITY
#define TRACER_HASH_TABLE_CAPACITY 512
#endif
#ifndef TRACER_SUBSCRIBER_BUFFER_SIZE
#define TRACER_SUBSCRIBER_BUFFER_SIZE (1 << 20)
#endif
// 0 = DROP, 1 = WAIT
#ifndef TRACER_QUEUE_FULL_POLICY
#define TRACER_QUEUE_FULL_POLICY 1
#endif
#ifndef TRACER_TCP_RECONNECT_MS
#define TRACER_TCP_RECONNECT_MS 1000
#endif
#ifndef TRACER_MAX_NDIM
#define TRACER_MAX_NDIM 8
#endif

typedef enum Result {
    SUCCESS = 0,
    SOCKET_INIT_FAILED = 1,
    SOCKET_CREATION_FAILED = 2,
    BIND_FAILED = 3,
    LISTEN_FAILED = 4,
    ACCEPT_FAILED = 5,
    CONNECT_FAILED = 6,
    CONNECT_WAITING = 7,
    INVALID_ADDRESS = 8,
    TIMEOUT = 9,
    THREAD_CREATION_FAILED = 10,
} Result;


typedef struct TcpConnection
{
    socket_t socket;
    struct addrinfo* addresses;
    struct addrinfo* picked_address;
} TcpConnection;

static Result socket_init();
static Result socket_listen(uint16_t port, int backlog, bool only_ipv4, bool only_localhost, socket_t* socket);
static Result socket_accept(socket_t listening_socket, int timeout, socket_t* socket);
static Result socket_connect( const char* addr, uint16_t port, TcpConnection* connection_socket);
static void socket_close(socket_t socket);
static Result socket_connect_blocking( const char* addr, uint16_t port, socket_t* connection_socket);
static void socket_close_connection(TcpConnection* connection);

#ifdef TRACING_IMPLEMENTATION

static Result
socket_init() {
#ifdef _WIN32
    WSADATA wsaData;
    if( WSAStartup( MAKEWORD( 2, 2 ), &wsaData ) != 0 )
    {
        return SOCKET_INIT_FAILED;
    }
#endif
    return SUCCESS;
}

static socket_t
__addrinfo_and_socket_for_family(uint16_t port, int ai_family, bool only_localhost, struct addrinfo** res)
{
    struct addrinfo hints;
    memset( &hints, 0, sizeof( hints ) );
    hints.ai_family = ai_family;
    hints.ai_socktype = SOCK_STREAM;
    if(!only_localhost)
    {
        hints.ai_flags = AI_PASSIVE;
    }

    char portbuf[32];
    snprintf( portbuf, sizeof(portbuf), "%" PRIu16, port );
    if( getaddrinfo( NULL, portbuf, &hints, res ) != 0 ) return -1;
    socket_t sock = socket( (*res)->ai_family, (*res)->ai_socktype, (*res)->ai_protocol );
    if (sock == -1) freeaddrinfo( *res );
    return sock;
}

static Result
socket_listen(uint16_t port, int backlog, bool only_ipv4, bool only_localhost, socket_t* socket) {
    socket_t sock = -1;
    struct addrinfo* res = NULL;

    if(!only_ipv4)
    {
        sock = __addrinfo_and_socket_for_family( port, AF_INET6, only_localhost, &res );
    }
    if (sock == -1)
    {
        // IPV6 protocol may not be available/is disabled. Try to create a socket
        // with the IPV4 protocol
        sock = __addrinfo_and_socket_for_family( port, AF_INET, only_localhost, &res );
        if( sock == -1 ) return SOCKET_CREATION_FAILED;
    }

#if defined _WIN32
    if (res->ai_family == AF_INET6) {
        unsigned long val = 0;
        setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&val, sizeof( val ) );
    }
#elif defined BSD
    int val;
    if (res->ai_family == AF_INET6) {
        val = 0;
        setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&val, sizeof( val ) );
    }
    val = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &val, sizeof( val ) );
#else
    int val = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &val, sizeof( val ) );
#endif
    if( bind(sock, res->ai_addr, res->ai_addrlen ) == -1 ) { freeaddrinfo( res ); socket_close(sock); return BIND_FAILED; }
    if( listen(sock, backlog ) == -1 ) { freeaddrinfo( res ); socket_close(sock); return LISTEN_FAILED; }
    freeaddrinfo( res );

    *socket = sock;
    return SUCCESS;
}

static Result
socket_accept(socket_t listening_socket, int timeout, socket_t* socket)
{
    struct sockaddr_storage remote;
    socklen_t sz = sizeof(remote);

    struct pollfd fd;
    fd.fd = listening_socket;
    fd.events = POLLIN;

    if(poll(&fd, 1, timeout) > 0)
    {
        socket_t sock = accept(listening_socket, (struct sockaddr*)&remote, &sz);
        if( sock == -1 ) return ACCEPT_FAILED;

#if defined __APPLE__
        int val = 1;
        setsockopt(sock, SOL_SOCKET, SO_NOSIGPIPE, &val, sizeof( val ) );
#endif
        *socket = sock;
        return SUCCESS;
    }
    else
    {
        return TIMEOUT;
    }
}

static void
socket_close(socket_t socket)
{
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

static Result
socket_connect( const char* addr, uint16_t port, TcpConnection* connection)
{
    if(connection->picked_address)
    {
        const int c = connect(connection->socket, connection->picked_address->ai_addr, connection->picked_address->ai_addrlen );
        if( c == -1 )
        {
#if defined _WIN32
            const int err = WSAGetLastError();
            if( err == WSAEALREADY || err == WSAEINPROGRESS ) return CONNECT_WAITING;
            if( err != WSAEISCONN )
            {
                freeaddrinfo( connection->addresses );
                closesocket( connection->socket );
                connection->addresses = 0;
                connection->picked_address = 0;
                return CONNECT_FAILED;
            }
#else
            const int err = errno;
            if( err == EALREADY || err == EINPROGRESS ) return CONNECT_WAITING;
            if( err != EISCONN )
            {
                freeaddrinfo( connection->addresses );
                close( connection->socket );
                connection->addresses = 0;
                connection->picked_address = 0;
                return CONNECT_FAILED;
            }
#endif
        }

#if defined _WIN32
        u_long nonblocking = 0;
        ioctlsocket( connection->socket, FIONBIO, &nonblocking );
#else
        int flags = fcntl( connection->socket, F_GETFL, 0 );
        fcntl( connection->socket, F_SETFL, flags & ~O_NONBLOCK );
#endif
        freeaddrinfo( connection->addresses );
        connection->addresses = 0;
        connection->picked_address = 0;
        return SUCCESS;
    }

    struct addrinfo hints;
    struct addrinfo *res, *ptr;

    memset( &hints, 0, sizeof( hints ) );
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char portbuf[32];
    snprintf( portbuf, sizeof(portbuf), "%" PRIu16, port );

    if( getaddrinfo( addr, portbuf, &hints, &res ) != 0 ) return INVALID_ADDRESS;
    socket_t sock = 0;
    for( ptr = res; ptr; ptr = ptr->ai_next )
    {
        if( ( sock = socket( ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol ) ) == -1 ) continue;
#if defined __APPLE__
        int val = 1;
        setsockopt( sock, SOL_SOCKET, SO_NOSIGPIPE, &val, sizeof( val ) );
#endif
#if defined _WIN32
        u_long nonblocking = 1;
        ioctlsocket( sock, FIONBIO, &nonblocking );
#else
        int flags = fcntl( sock, F_GETFL, 0 );
        fcntl( sock, F_SETFL, flags | O_NONBLOCK );
#endif
        if( connect( sock, ptr->ai_addr, ptr->ai_addrlen ) == 0 )
        {
            break;
        }
        else
        {
#if defined _WIN32
            const int err = WSAGetLastError();
            if( err != WSAEWOULDBLOCK )
            {
                closesocket( sock );
                continue;
            }
#else
            if( errno != EINPROGRESS )
            {
                close( sock );
                continue;
            }
#endif
        }

        connection->addresses = res;
        connection->picked_address = ptr;
        connection->socket = sock;
        return CONNECT_WAITING;
    }
    freeaddrinfo( res );
    if( !ptr ) return CONNECT_FAILED;

#if defined _WIN32
    u_long nonblocking = 0;
    ioctlsocket( sock, FIONBIO, &nonblocking );
#else
    int flags = fcntl( sock, F_GETFL, 0 );
    fcntl( sock, F_SETFL, flags & ~O_NONBLOCK );
#endif

    connection->socket = sock;
    connection->addresses = 0;
    connection->picked_address = 0;
    return SUCCESS;
}

static void
socket_close_connection(TcpConnection* connection) {
    socket_close(connection->socket);
    if (connection->addresses) {
        freeaddrinfo( connection->addresses );
    }
    connection->addresses = 0;
    connection->picked_address = 0;
}

static Result
socket_connect_blocking( const char* addr, uint16_t port, socket_t* connection_socket)
{
    struct addrinfo hints;
    struct addrinfo *res, *ptr;

    memset( &hints, 0, sizeof( hints ) );
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char portbuf[32];
    snprintf( portbuf, sizeof(portbuf), "%" PRIu16, port );

    if( getaddrinfo( addr, portbuf, &hints, &res ) != 0 ) return INVALID_ADDRESS;
    socket_t sock = 0;
    for( ptr = res; ptr; ptr = ptr->ai_next )
    {
        if( ( sock = socket( ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol ) ) == -1 ) continue;
#if defined __APPLE__
        int val = 1;
        setsockopt( sock, SOL_SOCKET, SO_NOSIGPIPE, &val, sizeof( val ) );
#endif
        if( connect( sock, ptr->ai_addr, ptr->ai_addrlen ) == -1 )
        {
#ifdef _WIN32
            closesocket( sock );
#else
            close( sock );
#endif
            continue;
        }
        break;
    }
    freeaddrinfo( res );
    if( !ptr ) return CONNECT_FAILED;

    *connection_socket = sock;
    return SUCCESS;
}


// Threading
#ifdef _WIN32
#define THREAD_PROC(x) DWORD WINAPI x(LPVOID data)
#else
#define THREAD_PROC(x) void* x(void* data)
#endif // _WIN32

typedef struct Thread
{
#ifdef _WIN32
    HANDLE handle;
#else
    pthread_t thread;
#endif
} Thread;

typedef THREAD_PROC(ThreadProc);

static Result create_thread(ThreadProc proc, void* user_data, Thread* thread) {
#ifdef _WIN32
    thread->handle = CreateThread(0, 0, proc, user_data, 0, 0);
    if (thread->handle == NULL) {
        return THREAD_CREATION_FAILED;
    }
#else
    int result = pthread_create(&thread->thread, 0, proc, user_data);
    if (result != 0) {
        return THREAD_CREATION_FAILED;
    }
#endif

    return SUCCESS;
}

static void
join_thread(Thread* thread) {
#ifdef _WIN32
    WaitForSingleObject(thread->handle, INFINITE);
    CloseHandle(thread->handle);
#else
    pthread_join(thread->thread, 0);
#endif
}

typedef atomic_uint Futex;

static inline void futex_init(Futex* futex, uint32_t initial_value) {
    atomic_store_explicit(futex, initial_value, memory_order_relaxed);
}

// Could return false on timeout (not implemented yet)
static inline bool futex_wait(Futex* futex, uint32_t expected) {
#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4090 )
    return WaitOnAddress(futex, &expected, 4, INFINITE) == TRUE;
#pragma warning( pop )
#elif defined(__APPLE__)
    while (true) {
        int ret = os_sync_wait_on_address(futex, expected, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE);
        if ((ret < 0) && ((errno == EINTR) || (errno == EFAULT))) {
           continue;
        }
        break;
    }
    return true;
#else
    while (true) {
        if (atomic_load_explicit(futex, memory_order_relaxed) != expected) {
            break;
        }

        int ret = syscall(SYS_futex, futex, FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expected, NULL, NULL, 0);
        if ((ret < 0) && (errno == EINTR)) {
           continue;
        }
        break;
    }
    return true;
#endif
}

// Returns true if we can guarantee a thread was awaked
static inline bool futex_wake(Futex* futex) {
#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4090 )
    WakeByAddressSingle(futex);
#pragma warning( pop )
    return false;
#elif defined(__APPLE__)
    int ret = os_sync_wake_by_address_any(futex, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
    if (ret < 0) {
        return false;
    }
    return true;
#else
    return syscall(SYS_futex, futex, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1) > 0;
#endif
}

static inline void futex_wake_all(Futex* futex) {
#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4090 )
    WakeByAddressAll(futex);
#pragma warning( pop )
#elif defined(__APPLE__)
    os_sync_wake_by_address_all(futex, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
#else
    syscall(SYS_futex, futex, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, INT32_MAX);
#endif
}

// Mutex
#define MUTEX_UNLOCKED  ((uint32_t)0)
#define MUTEX_LOCKED    ((uint32_t)1)  // locked, no other threads waiting
#define MUTEX_CONTENDED ((uint32_t)2)  // locked, and other threads waiting (contended)

typedef struct Mutex {
    Futex futex;
} Mutex;

static inline void mutex_init(Mutex* mutex) {
    futex_init(&mutex->futex, MUTEX_UNLOCKED);
}

static inline bool mutex_try_lock(Mutex* mutex) {
    uint32_t expected = MUTEX_UNLOCKED;
    return atomic_compare_exchange_strong_explicit(&mutex->futex, &expected, MUTEX_LOCKED, memory_order_acquire, memory_order_relaxed);
}

static uint32_t _mutex_spin(Mutex* mutex) {
    int spin = 100;
    while (true) {
        // We only use `load` (and not `swap` or `compare_exchange`)
        // while spinning, to be easier on the caches.
        uint32_t state = atomic_load_explicit(&mutex->futex, memory_order_relaxed);

        // We stop spinning when the mutex is UNLOCKED,
        // but also when it's CONTENDED.
        if (state != MUTEX_LOCKED || spin == 0) {
            return state;
        }
        SpinlockHint();
        spin--;
    }
}

static void _mutex_lock_contended(Mutex* mutex) {
    // Spin first to speed things up if the lock is released quickly.
    uint32_t state = _mutex_spin(mutex);

    // If it's unlocked now, attempt to take the lock
    // without marking it as contended.
    if (state == MUTEX_UNLOCKED) {
        uint32_t expected = MUTEX_UNLOCKED;
        if (atomic_compare_exchange_strong_explicit(&mutex->futex, &expected, MUTEX_LOCKED, memory_order_acquire, memory_order_relaxed)) {
            return; // Locked!
        }
        state = expected;
    }

    while (true) {
        // Put the lock in contended state.
        // We avoid an unnecessary write if it's already set to CONTENDED,
        // to be friendlier for the caches.
        if (state != MUTEX_CONTENDED && atomic_exchange_explicit(&mutex->futex, MUTEX_CONTENDED, memory_order_acquire) == MUTEX_UNLOCKED) {
            // We changed it from UNLOCKED to CONTENDED, so we just successfully locked it.
            return;
        }

        // Wait for the futex to change state, assuming it is still CONTENDED.
        futex_wait(&mutex->futex, MUTEX_CONTENDED);

        // Spin again after waking up.
        state = _mutex_spin(mutex);
    }
}

static inline void mutex_lock(Mutex* mutex) {
    uint32_t expected = MUTEX_UNLOCKED;
    if (!atomic_compare_exchange_strong_explicit(&mutex->futex, &expected, MUTEX_LOCKED, memory_order_acquire, memory_order_relaxed)) {
        _mutex_lock_contended(mutex);
    }
}

static inline void mutex_unlock(Mutex* mutex) {
    if (atomic_exchange_explicit(&mutex->futex, MUTEX_UNLOCKED, memory_order_release) == MUTEX_CONTENDED) {
        // We only wake up one thread. When that thread locks the mutex, it
        // will mark the mutex as CONTENDED (see _mutex_lock_contended above),
        // which makes sure that any other waiting threads will also be
        // woken up eventually.
        futex_wake(&mutex->futex);
    }
}

#define RWLOCK_READ_LOCKED        ((uint32_t)1)
#define RWLOCK_MASK               ((uint32_t)((1 << 30) - 1))
#define RWLOCK_WRITE_LOCKED       RWLOCK_MASK
#define RWLOCK_DOWNGRADE          ((uint32_t)(RWLOCK_READ_LOCKED - RWLOCK_WRITE_LOCKED))
#define RWLOCK_MAX_READERS        ((uint32_t)(RWLOCK_MASK - 1))
#define RWLOCK_READERS_WAITING    ((uint32_t)(1 << 30))
#define RWLOCK_WRITERS_WAITING    ((uint32_t)1 << 31)

typedef struct RWLock {
    // The state consists of a 30-bit reader counter, a 'readers waiting' flag, and a 'writers waiting' flag.
    // Bits 0..30:
    //   0: Unlocked
    //   1..=0x3FFF_FFFE: Locked by N readers
    //   0x3FFF_FFFF: Write locked
    // Bit 30: Readers are waiting on this futex.
    // Bit 31: Writers are waiting on the writer_notify futex.
    Futex state;

    // The 'condition variable' to notify writers through.
    // Incremented on every signal.
    Futex writer_notify;
} RWLock;

static inline bool _rwlock_is_unlocked(uint32_t state) {
    return (state & RWLOCK_MASK) == 0;
}

static inline bool _rwlock_is_write_locked(uint32_t state) {
    return (state & RWLOCK_MASK) == RWLOCK_WRITE_LOCKED;
}

static inline bool _rwlock_has_readers_waiting(uint32_t state) {
    return (state & RWLOCK_READERS_WAITING) != 0;
}

static inline bool _rwlock_has_writers_waiting(uint32_t state) {
    return (state & RWLOCK_WRITERS_WAITING) != 0;
}

static inline bool _rwlock_is_read_lockable(uint32_t state) {
    // This also returns false if the counter could overflow if we tried to read lock it.
    //
    // We don't allow read-locking if there's readers waiting, even if the lock is unlocked
    // and there's no writers waiting. The only situation when this happens is after unlocking,
    // at which point the unlocking thread might be waking up writers, which have priority over readers.
    // The unlocking thread will clear the readers waiting bit and wake up readers, if necessary.
    return (state & RWLOCK_MASK) < RWLOCK_MAX_READERS && !_rwlock_has_readers_waiting(state) && !_rwlock_has_writers_waiting(state);
}

static inline bool _rwlock_is_read_lockable_after_wakeup(uint32_t state){
    // We make a special case for checking if we can read-lock _after_ a reader thread that went to
    // sleep has been woken up by a call to `downgrade`.
    //
    // `downgrade` will wake up all readers and place the lock in read mode. Thus, there should be
    // no readers waiting and the lock should be read-locked (not write-locked or unlocked).
    //
    // Note that we do not check if any writers are waiting. This is because a call to `downgrade`
    // implies that the caller wants other readers to read the value protected by the lock. If we
    // did not allow readers to acquire the lock before writers after a `downgrade`, then only the
    // original writer would be able to read the value, thus defeating the purpose of `downgrade`.
    return (state & RWLOCK_MASK) < RWLOCK_MAX_READERS
        && !_rwlock_has_readers_waiting(state)
        && !_rwlock_is_write_locked(state)
        && !_rwlock_is_unlocked(state);
}

static inline bool _rwlock_has_reached_max_readers(uint32_t state) {
    return (state & RWLOCK_MASK) == RWLOCK_MAX_READERS;
}


static inline void rwlock_init(RWLock* rwlock) {
    futex_init(&rwlock->state, 0);
    futex_init(&rwlock->writer_notify, 0);
}

static inline bool rwlock_try_read(RWLock* rwlock) {
    uint32_t prev = atomic_load_explicit(&rwlock->state, memory_order_acquire);
    while (true) {
        if (!_rwlock_is_read_lockable(prev)) {
            return false;
        }
        uint32_t desired = prev + RWLOCK_READ_LOCKED;
        if (atomic_compare_exchange_weak_explicit(&rwlock->state, &prev, desired, memory_order_acquire, memory_order_relaxed)) {
            return true;
        }
    }
}

// Spin for a while, but stop directly at the given condition.
// #[inline]
// fn spin_until(&self, f: impl Fn(Primitive) -> bool) -> Primitive {
//     let mut spin = 100; // Chosen by fair dice roll.
//     loop {
//         let state = self.state.load(Relaxed);
//         if f(state) || spin == 0 {
//             return state;
//         }
//         crate::hint::spin_loop();
//         spin -= 1;
//     }
// }

static inline uint32_t _rwlock_spin_write(RWLock* rwlock) {
    size_t spin = 100;
    while(true) {
        uint32_t state = atomic_load_explicit(&rwlock->state, memory_order_relaxed);

        // Stop spinning when it's unlocked or when there's waiting writers, to keep things somewhat fair.
        if ((_rwlock_is_unlocked(state) || _rwlock_has_writers_waiting(state)) || spin == 0) {
            return state;
        }
        SpinlockHint();
        spin--;
    }
}

static inline uint32_t _rwlock_spin_read(RWLock* rwlock) {
    size_t spin = 100;
    while(true) {
        uint32_t state = atomic_load_explicit(&rwlock->state, memory_order_relaxed);

        // Stop spinning when it's unlocked or when there's waiting writers, to keep things somewhat fair.
        if ((!_rwlock_is_write_locked(state) || _rwlock_has_readers_waiting(state) || _rwlock_has_writers_waiting(state)) || spin == 0) {
            return state;
        }
        SpinlockHint();
        spin--;
    }
}

static void _rwlock_read_contended(RWLock* rwlock) {
    bool has_slept = false;
    uint32_t state = _rwlock_spin_read(rwlock);

    while (true) {
        // If we have just been woken up, first check for a `downgrade` call.
        // Otherwise, if we can read-lock it, lock it.
        if ((has_slept && _rwlock_is_read_lockable_after_wakeup(state)) || _rwlock_is_read_lockable(state)) {
            uint32_t desired = state + RWLOCK_READ_LOCKED;
            if (atomic_compare_exchange_weak_explicit(&rwlock->state, &state, desired, memory_order_acquire, memory_order_relaxed)) {
                return;
            } else {
                continue;
            }
        }

        // Check for overflow.
        // assert(!_rwlock_has_reached_max_readers(state), "too many active read locks on RwLock");

        // Make sure the readers waiting bit is set before we go to sleep.
        if (!_rwlock_has_readers_waiting(state)) {
            uint32_t expected = state;
            uint32_t desired = state | RWLOCK_READERS_WAITING;
            if (!atomic_compare_exchange_strong_explicit(&rwlock->state, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
                state = expected;
                continue;
            }
        }

        // Wait for the state to change.
        futex_wait(&rwlock->state, state | RWLOCK_READERS_WAITING);
        has_slept = true;

        // Spin again after waking up.
        state = _rwlock_spin_read(rwlock);
    }
}

static inline void rwlock_read(RWLock* rwlock) {
    uint32_t state = atomic_load_explicit(&rwlock->state, memory_order_relaxed);
    if (!_rwlock_is_read_lockable(state) || !atomic_compare_exchange_weak_explicit(&rwlock->state, &state, state + RWLOCK_READ_LOCKED, memory_order_acquire, memory_order_relaxed))
    {
        // Do contended acquire if not lockable or cannot acquire immediately.
        _rwlock_read_contended(rwlock);
    }
}


// This wakes one writer and returns true if we woke up a writer that was
// blocked on futex_wait.
//
// If this returns false, it might still be the case that we notified a
// writer that was about to go to sleep.
static inline bool _rwlock_wake_writer(RWLock* rwlock) {
    atomic_fetch_add_explicit(&rwlock->writer_notify, 1, memory_order_release);
    return futex_wake(&rwlock->writer_notify);
    // Note that some platforms don't tell us whether they woke
    // up any threads or not, and always return `false` here. That still
    // results in correct behavior: it just means readers get woken up as
    // well in case both readers and writers were waiting.
}

static void _rwlock_wake_writer_or_readers(RWLock* rwlock, uint32_t state) {
    // assert(_rwlock_is_unlocked(state));

    // The readers waiting bit might be turned on at any point now,
    // since readers will block when there's anything waiting.
    // Writers will just lock the lock though, regardless of the waiting bits,
    // so we don't have to worry about the writer waiting bit.
    //
    // If the lock gets locked in the meantime, we don't have to do
    // anything, because then the thread that locked the lock will take
    // care of waking up waiters when it unlocks.

    // If only writers are waiting, wake one of them up.
    if (state == RWLOCK_WRITERS_WAITING) {
        uint32_t expected = state;
        uint32_t desired = 0;
        if (atomic_compare_exchange_strong_explicit(&rwlock->state, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            _rwlock_wake_writer(rwlock);
            return;
        } else {
            // Maybe some readers are now waiting too. So, continue to the next `if`.
            state = expected;
        }
    }

    // If both writers and readers are waiting, leave the readers waiting
    // and only wake up one writer.
    if (state == RWLOCK_READERS_WAITING + RWLOCK_WRITERS_WAITING) {
        uint32_t expected = state;
        uint32_t desired = RWLOCK_READERS_WAITING;
        if (!atomic_compare_exchange_strong_explicit(&rwlock->state, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            // The lock got locked. Not our problem anymore.
            return;
        }

        if (_rwlock_wake_writer(rwlock)) {
            return;
        }

        // No writers were actually blocked on futex_wait, so we continue
        // to wake up readers instead, since we can't be sure if we notified a writer.
        state = RWLOCK_READERS_WAITING;
    }

    // If readers are waiting, wake them all up.
    if (state == RWLOCK_READERS_WAITING) {
        uint32_t expected = state;
        uint32_t desired = 0;
        if (atomic_compare_exchange_strong_explicit(&rwlock->state, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            futex_wake_all(&rwlock->state);
        }
    }
}

static inline void rwlock_read_unlock(RWLock* rwlock) {
    uint32_t state = atomic_fetch_sub_explicit(&rwlock->state, RWLOCK_READ_LOCKED, memory_order_release) - RWLOCK_READ_LOCKED;

    // It's impossible for a reader to be waiting on a read-locked RwLock,
    // except if there is also a writer waiting.
    // debug_assert(!_rwlock_has_readers_waiting(state) || _rwlock_has_writers_waiting(state));

    // Wake up a writer if we were the last reader and there's a writer waiting.
    if (_rwlock_is_unlocked(state) && _rwlock_has_writers_waiting(state)) {
        _rwlock_wake_writer_or_readers(rwlock, state);
    }
}

static inline bool rwlock_try_write(RWLock* rwlock) {
    uint32_t prev = atomic_load_explicit(&rwlock->state, memory_order_acquire);
    while (true) {
        if (!_rwlock_is_unlocked(prev)) {
            return false;
        }
        uint32_t desired = prev + RWLOCK_WRITE_LOCKED;
        if (atomic_compare_exchange_weak_explicit(&rwlock->state, &prev, desired, memory_order_acquire, memory_order_relaxed)) {
            return true;
        }
    }
}

static void _rwlock_write_contended(RWLock* rwlock) {
    uint32_t state = _rwlock_spin_write(rwlock);
    uint32_t other_writers_waiting = 0;

    while (true) {
        // If it's unlocked, we try to lock it.
        if (_rwlock_is_unlocked(state)) {
            if (atomic_compare_exchange_weak_explicit(&rwlock->state, &state, state | RWLOCK_WRITE_LOCKED | other_writers_waiting, memory_order_acquire, memory_order_relaxed)) {
                return; // Locked!
            } else {
                continue;
            }
        }

        // Set the waiting bit indicating that we're waiting on it.
        if (!_rwlock_has_writers_waiting(state)) {
            uint32_t expected = state;
            uint32_t desired = state | RWLOCK_WRITERS_WAITING;
            if (!atomic_compare_exchange_strong_explicit(&rwlock->state, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
                // The lock got locked. Not our problem anymore.
                state = expected;
                continue;
            }
        }

        // Other writers might be waiting now too, so we should make sure
        // we keep that bit on once we manage lock it.
        other_writers_waiting = RWLOCK_WRITERS_WAITING;

        // Examine the notification counter before we check if `state` has changed,
        // to make sure we don't miss any notifications.
        uint32_t seq = atomic_load_explicit(&rwlock->writer_notify, memory_order_acquire);

        // Don't go to sleep if the lock has become available,
        // or if the writers waiting bit is no longer set.
        state = atomic_load_explicit(&rwlock->state, memory_order_relaxed);
        if (_rwlock_is_unlocked(state) || !_rwlock_has_writers_waiting(state)) {
            continue;
        }

        // Wait for the state to change.
        futex_wait(&rwlock->writer_notify, seq);

        // Spin again after waking up.
        state = _rwlock_spin_write(rwlock);
    }
}

static inline void rwlock_write(RWLock* rwlock) {
    uint32_t expected = 0;
    if (!atomic_compare_exchange_weak_explicit(&rwlock->state, &expected, RWLOCK_WRITE_LOCKED, memory_order_acquire, memory_order_relaxed)) {
        _rwlock_write_contended(rwlock);
    }
}

static inline void rwlock_write_unlock(RWLock* rwlock) {
    uint32_t state = atomic_fetch_sub_explicit(&rwlock->state, RWLOCK_WRITE_LOCKED, memory_order_release) - RWLOCK_WRITE_LOCKED;
    // debug_assert(_rwlock_is_unlocked(state));

    if (_rwlock_has_writers_waiting(state) || _rwlock_has_readers_waiting(state)) {
        _rwlock_wake_writer_or_readers(rwlock, state);
    }
}

static inline void downgrade(RWLock* rwlock) {
    // Removes all write bits and adds a single read bit.
    uint32_t state = atomic_fetch_add_explicit(&rwlock->state, RWLOCK_DOWNGRADE, memory_order_release);

    // debug_assert(_rwlock_is_write_locked(state), "RwLock must be write locked to call `downgrade`");

    if (_rwlock_has_readers_waiting(state)) {
        // Since we had the exclusive lock, nobody else can unset this bit.
        atomic_fetch_sub_explicit(&rwlock->state, RWLOCK_READERS_WAITING, memory_order_relaxed);
        futex_wake_all(&rwlock->state);
    }
}

// Ring-mapped buffer from: https://gist.github.com/mmozeiko/3b09a340f3c53e5eaed699a1aea95250
static void* alloc_ring_mapped_buffer(size_t Size)
{
    void* data;
#if defined(_WIN32)
	SYSTEM_INFO SystemInfo;
	GetSystemInfo(&SystemInfo);

	const size_t PageSize = SystemInfo.dwPageSize;
	Size = (Size + PageSize - 1) & ~(PageSize - 1);

	char* Placeholder1 = (char*)VirtualAlloc2(NULL, NULL, 2 * Size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0);
	char* Placeholder2 = (char*)Placeholder1 + Size;
	ASSERT(Placeholder1 && "failed to reserve placeholder");

	BOOL Ok = VirtualFree(Placeholder1, Size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
	ASSERT(Ok && "failed to split reservation into two placeholders");

	HANDLE Section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)(Size >> 32), (DWORD)Size, NULL);
	ASSERT(Section && "failed to create mapping");

	void* View1 = MapViewOfFile3(Section, NULL, Placeholder1, 0, Size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0);
	ASSERT(View1 && "failed to map first half of mapping");

	void* View2 = MapViewOfFile3(Section, NULL, Placeholder2, 0, Size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0);
	ASSERT(View2 && "failed to map second half of mapping");

	CloseHandle(Section);
	VirtualFree(Placeholder1, 0, MEM_RELEASE);
	VirtualFree(Placeholder2, 0, MEM_RELEASE);

	data = View1;

#elif defined(__APPLE__)

	Size = mach_vm_round_page(Size);

	mach_port_t Task = mach_task_self();

	mach_vm_address_t Address;
	int AllocateOk = mach_vm_allocate(Task, &Address, 2 * Size, VM_FLAGS_ANYWHERE);
	ASSERT(AllocateOk == 0 && "failed to allocate memory");

	int Mapping1 = mach_vm_allocate(Task, &Address, Size, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE);
	ASSERT(Mapping1 == 0 && "failed to map first half of mapping");

	const vm_prot_t PageProtection = VM_PROT_READ | VM_PROT_WRITE;

	mach_port_t MemoryPort;
	mach_vm_size_t MemorySize = Size;
	int PortOk = mach_make_memory_entry_64(Task, &MemorySize, Address, PageProtection, &MemoryPort, MACH_PORT_NULL);
	ASSERT(PortOk == 0 && "failed to create mach port");

	mach_vm_address_t Address2 = Address + Size;
	int Mapping2 = mach_vm_map(Task, &Address2, Size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, MemoryPort, 0, FALSE, PageProtection, PageProtection, VM_INHERIT_NONE);
	ASSERT(Mapping2 == 0 && "failed to map second half of mapping");

	mach_port_deallocate(Task, MemoryPort);

	data = (void*)Address;

#else

	const size_t PageSize = sysconf(_SC_PAGESIZE);
	Size = (Size + PageSize - 1) & ~(PageSize - 1);

	int MemFd = syscall(__NR_memfd_create, "MagicRingBuffer", FD_CLOEXEC);
	ASSERT(MemFd > 0 && "failed to create memfd");

	int Ok = ftruncate(MemFd, Size);
	ASSERT(Ok == 0 && "failed to set memfd size");

	char* Base = (char*)mmap(NULL, 2 * Size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	ASSERT(Base != MAP_FAILED && "failed to create memory mapping");

	void* Mapped1 = mmap(Base + 0 * Size, Size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, MemFd, 0);
	ASSERT(Mapped1 != MAP_FAILED && "failed to map first half of mapping");

	void* Mapped2 = mmap(Base + 1 * Size, Size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, MemFd, 0);
	ASSERT(Mapped2 != MAP_FAILED && "failed to map second half of mapping");

	close(MemFd);

	data = Base;
#endif
    return data;
}

static void free_ring_mapped_buffer(void* ptr, size_t size) {
#if defined(_WIN32)
	UnmapViewOfFileEx((char*)ptr, 0);
	UnmapViewOfFileEx((char*)ptr + size, 0);
#elif defined(__APPLE__)
	mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)ptr, 2 * size);
#else
	munmap(ptr, 2 * size);
#endif
}

// Pre-allocated MPSC ringbuffer with wait and drop semantics
//
// Notes:
// - If we allow concurrent allocs before commits, we need to allow out-of-order commit as well,
//   otherwise you run into issues where you need to wait for previous writers to be done with
//   their allocations before you are allowed to commit.
//
//   For example the worst case scenario would be something like 1GiB alloc followed by 8 bytes alloc.
//   In this case the 8 bytes alloc would have to wait for the large 1GiB memcpy before going through.
//
// The underlying issue is that to have allocation and consumption be linear we have to commit in order.
//
// Actually the issue is mostly with the linear shared allocator, because it can be dealloced only in alloc
// order, regardless of commit order.
// We either use a different allocator or implement some sort of delegated deallocation (which requires a log).
//
// Potential solutions:
// - Allow early return from commit by deferring actual commit to other writers
//   -> would need a separate data structure to act as a log of uncommitted
//   -> need to keep this sorted, or traverse in sorted order after every commit
//   -> can tune number of "in-flight" allocations and potentially return a failure to try_alloc
//   -> does not solve the general case when we run out of log slots (or need to allow unbounded growth)
//      -> not bounded by the number of concurrent producers because you can run into limits even with 2 threads
//          -> A allocs 1 GiB and holds the first commit slot
//          -> B can alloc N small elements while A is holding the slot.
//   -> need to make a "ticket" style API where you reserve a slot in the log
// - Truth is that logging is cheap, we expect to be consumer bound in most cases, so we can allow these to block on commit in most-cases.
//
// Thoughts:
// - maybe committed could be fully writer-owned? Ring-buffer is defined by allocated and consumed while
//   committed ranges are enqueued to a writer. The writer then takes responsability of bumping consumed
//   by keeping track of used chunks (fragmented to linear). After the chunks are used the writer
//   could also manage a "free-list" of chunks to deallocate and to bump the consumed index. If we force
//   a minimum alloc size the free list could also be stored by repurposing allocations (can reuse the memory
//   after it's consumed).
//   -> need an enqueue side channel (basically a fixed element-size mpsc)
// - a general purpose allocator also solves this (this is what we do in python basically), but has no bounded cost guarantee
//
// - I want to avoid the weird edge case where a slow producer causes us to drop messages even if we would have memory to store them, we just
//   don't have the side channel memory to store it.
// - SOLUTION:
//    - store the side-channel data inline with the payload, we would likely anyways need this to do message framing.
//    - this also allows out-of-order commit because the producer will just mark the entry
//    - do we still mutex wait on the counters, or is it even better to directly wait on the item marker?
//    - this has to be the better design, no side-channel, simple metadata,
//    - ISSUE: we have to clear the whole buffer after finishing a consume operations because we don't know where the next doorbell might be.
typedef struct MpscRingBuffer
{
    Mutex producers_mutex;   // Lock for atomic concurrent allocation and doorbell initialization
    uint8_t __padding0[CACHE_LINE_SIZE - sizeof(Mutex)];

    Futex produced_offset;   // Producers write, consumer reads and waits
    uint8_t __padding1[CACHE_LINE_SIZE - sizeof(Futex)];

    Futex consumed_offset;  // Producers reads and waits, consumer writes
    uint8_t __padding2[CACHE_LINE_SIZE - sizeof(Futex)];

    uint8_t* ring_buffer;
    size_t size;
    uint32_t mask;
} MpscRingBuffer;

#define MPSC_HEADER_TOTAL_SIZE 16
#define MPSC_HEADER_DOORBELL_OFFSET 0
#define MPSC_HEADER_PAYLOAD_SIZE_OFFSET 8

// Doorbell values (per-entry, in header at MPSC_HEADER_DOORBELL_OFFSET)
#define MPSC_DOORBELL_EMPTY            ((uint32_t)0) // not committed
#define MPSC_DOORBELL_COMMITTED        ((uint32_t)1) // committed
#define MPSC_DOORBELL_CONSUMER_WAITING ((uint32_t)2) // consumer waiting for commit

#define MPSC_ALLOCATION_ALIGNMENT_BITS 4
#define MPSC_ALLOCATION_ALIGNMENT (1 << MPSC_ALLOCATION_ALIGNMENT_BITS)
#define MPSC_ALLOCATION_ALIGNMENT_MASK (MPSC_ALLOCATION_ALIGNMENT - 1)
#define MPSC_RING_BUFFER_MIN_SIZE 4096

static inline
bool is_pow2(size_t n) {
    return (n & (n - 1)) == 0;
}

static inline
size_t align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

static void
mpsc_ring_buffer_create(MpscRingBuffer* mpsc, size_t size) {
    mutex_init(&mpsc->producers_mutex);
    mpsc->produced_offset = 0;
    mpsc->consumed_offset = 0;

    // Buffer must be power of two.
    ASSERT(is_pow2(size));

    // We have 32 bits of addressing due to futex being 4 bytes.
    // We must leave a bit free to distinguish the queue full and queue empty case.
    // We index 16 byte blocks.
    // We thus have 2**31 * 16 = 32 GiB of max buffer size.
    ASSERT((uint64_t)size <= ((uint64_t)(32ull * 1024ull * 1024ull * 1024ull)));

    size = size < MPSC_RING_BUFFER_MIN_SIZE ? MPSC_RING_BUFFER_MIN_SIZE : size;

    mpsc->ring_buffer = (uint8_t*)alloc_ring_mapped_buffer(size);
    mpsc->size = size;

    mpsc->mask = (uint32_t)((size >> MPSC_ALLOCATION_ALIGNMENT_BITS) - 1);
}

static void
mpsc_ring_buffer_destroy(MpscRingBuffer* mpsc) {
    free_ring_mapped_buffer((void*)mpsc->ring_buffer, mpsc->size);

    mpsc->produced_offset = 0;
    mpsc->consumed_offset = 0;
    mpsc->consumed_offset = 0;

    mpsc->ring_buffer = 0;
    mpsc->size = 0;
}

static uint8_t*
mpsc_ring_buffer_try_reserve_write(MpscRingBuffer* mpsc, size_t size) {
    // Zero allocs or allocs that are bigger than the max payload size are not valid.
    if (size == 0 || size > (mpsc->size - MPSC_HEADER_TOTAL_SIZE)) {
        return NULL;
    }

    uint32_t alloc = (align_up(size, MPSC_ALLOCATION_ALIGNMENT) + MPSC_HEADER_TOTAL_SIZE) >> MPSC_ALLOCATION_ALIGNMENT_BITS;
    uint32_t capacity = mpsc->size >> MPSC_ALLOCATION_ALIGNMENT_BITS;

    // Try to alloc
    mutex_lock(&mpsc->producers_mutex);

    uint32_t produced = atomic_load_explicit(&mpsc->produced_offset, memory_order_relaxed);
    uint32_t consumed = atomic_load_explicit(&mpsc->consumed_offset, memory_order_acquire);

    // Check for space in buffer
    if (capacity - (produced - consumed) < alloc) {
        mutex_unlock(&mpsc->producers_mutex);
        return NULL;
    }

    uint8_t* start = mpsc->ring_buffer + ((size_t)(produced & mpsc->mask) << MPSC_ALLOCATION_ALIGNMENT_BITS);

    // Initialize header: clear doorbell and write payload size.
    atomic_store_explicit((Futex*)(start + MPSC_HEADER_DOORBELL_OFFSET), MPSC_DOORBELL_EMPTY, memory_order_relaxed);
    *(size_t*)(start + MPSC_HEADER_PAYLOAD_SIZE_OFFSET) = size;

    // Make the entry visible to the consumer.
    atomic_store_explicit(&mpsc->produced_offset, produced + alloc, memory_order_release);

    mutex_unlock(&mpsc->producers_mutex);

    // Return pointer to payload
    return start + MPSC_HEADER_TOTAL_SIZE;
}

static uint8_t*
mpsc_ring_buffer_wait_reserve_write(MpscRingBuffer* mpsc, size_t size) {
    // Zero allocs or allocs that are bigger than the max payload size are not valid.
    if (size == 0 || size > (mpsc->size - MPSC_HEADER_TOTAL_SIZE)) {
        return NULL;
    }

    uint32_t alloc = (align_up(size, MPSC_ALLOCATION_ALIGNMENT) + MPSC_HEADER_TOTAL_SIZE) >> MPSC_ALLOCATION_ALIGNMENT_BITS;
    uint32_t capacity = mpsc->size >> MPSC_ALLOCATION_ALIGNMENT_BITS;

    while (true) {
        mutex_lock(&mpsc->producers_mutex);
        uint32_t produced = atomic_load_explicit(&mpsc->produced_offset, memory_order_relaxed);
        uint32_t consumed = atomic_load_explicit(&mpsc->consumed_offset, memory_order_acquire);

        // If not enough space available wait for consumed to increase
        while (capacity - (produced - consumed) < alloc) {
            // Release the lock before going to sleep
            mutex_unlock(&mpsc->producers_mutex);

            // Wait for changes to consumed offset
            futex_wait(&mpsc->consumed_offset, consumed);

            // Re-acquire the lock before reading offsets again
            mutex_lock(&mpsc->producers_mutex);

            // Re-read allocated and consumed offsets
            produced = atomic_load_explicit(&mpsc->produced_offset, memory_order_relaxed);
            consumed = atomic_load_explicit(&mpsc->consumed_offset, memory_order_acquire);
        }

        uint8_t* start = mpsc->ring_buffer + ((size_t)(produced & mpsc->mask) << MPSC_ALLOCATION_ALIGNMENT_BITS);

        // Initialize header: clear doorbell and write payload size.
        atomic_store_explicit((Futex*)(start + MPSC_HEADER_DOORBELL_OFFSET), MPSC_DOORBELL_EMPTY, memory_order_relaxed);
        *(size_t*)(start + MPSC_HEADER_PAYLOAD_SIZE_OFFSET) = size;

        // Make the entry visible to the consumer.
        atomic_store_explicit(&mpsc->produced_offset, produced + alloc, memory_order_release);

        mutex_unlock(&mpsc->producers_mutex);

        // Return pointer to payload
        return start + MPSC_HEADER_TOTAL_SIZE;
    }
}

static void
mpsc_ring_buffer_commit_write(MpscRingBuffer* mpsc, uint8_t* alloc) {
    uint8_t* header = alloc - MPSC_HEADER_TOTAL_SIZE;

    // Swap doorbell to COMMITTED. If the consumer was waiting on this entry
    // (prev == CONSUMER_WAITING), wake the doorbell futex.
    Futex* doorbell = (Futex*)(header + MPSC_HEADER_DOORBELL_OFFSET);
    uint32_t prev = atomic_exchange_explicit(doorbell, MPSC_DOORBELL_COMMITTED, memory_order_release);
    if (prev == MPSC_DOORBELL_CONSUMER_WAITING) {
        futex_wake((Futex*)doorbell);
    } else {
        // Consumer is not waiting on doorbell, he might be waiting on visible data.
        futex_wake(&mpsc->produced_offset);
    }
}

static size_t
mpsc_ring_buffer_lock_acquire_read(MpscRingBuffer* mpsc, uint8_t** data) {
    uint32_t visible = atomic_load_explicit(&mpsc->produced_offset, memory_order_relaxed);
    uint32_t consumed = atomic_load_explicit(&mpsc->consumed_offset, memory_order_relaxed);

    // Sleep while no visible entries.
    // visible_offset is only advanced after the header is initialized (doorbell=0, size written),
    // so any entry the consumer can see has a valid header.
    while (visible == consumed) {
        futex_wait(&mpsc->produced_offset, visible);
        visible = atomic_load_explicit(&mpsc->produced_offset, memory_order_relaxed);
    }

    // There is a visible entry with an initialized header, check if it's committed.
    uint8_t* start = mpsc->ring_buffer + ((size_t)(consumed & mpsc->mask) << MPSC_ALLOCATION_ALIGNMENT_BITS);
    Futex* doorbell = (Futex*)start;

    // Wait until doorbell is committed.
    // acquire ordering synchronizes with the producer's release exchange in commit_write,
    // making the payload and header size visible to this thread.
    while (true) {
        uint32_t prev = atomic_exchange_explicit(doorbell, MPSC_DOORBELL_CONSUMER_WAITING, memory_order_acquire);
        if (prev == MPSC_DOORBELL_COMMITTED) break;

        // Doorbell is now CONSUMER_WAITING. Sleep until the producer swaps it to COMMITTED.
        futex_wait(doorbell, MPSC_DOORBELL_CONSUMER_WAITING);
    }

    // Return the data
    size_t size = *(size_t*)(start + MPSC_HEADER_PAYLOAD_SIZE_OFFSET);
    *data = start + MPSC_HEADER_TOTAL_SIZE;

    return size;
}

static void
mpsc_ring_buffer_lock_release_read(MpscRingBuffer* mpsc, size_t size) {
    // Increment the counter
    uint32_t consumed = atomic_load_explicit(&mpsc->consumed_offset, memory_order_relaxed);
    uint32_t alloc = (align_up(size, MPSC_ALLOCATION_ALIGNMENT) + MPSC_HEADER_TOTAL_SIZE) >> MPSC_ALLOCATION_ALIGNMENT_BITS;
    atomic_store_explicit(&mpsc->consumed_offset, consumed + alloc, memory_order_release);

    // Wake any potential waiting reader
    futex_wake_all(&mpsc->consumed_offset);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

typedef enum TraceType {
    TRACE_TYPE_NONE    = 0,
    TRACE_TYPE_I64     = 1,
    TRACE_TYPE_F64     = 2,
    TRACE_TYPE_STR     = 3,
    TRACE_TYPE_BYTES   = 7,
    TRACE_TYPE_NDARRAY = 8,
    TRACE_TYPE_I32     = 9,
    TRACE_TYPE_F32     = 10,
    TRACE_TYPE_U64     = 11,
    TRACE_TYPE_U32     = 12,
    TRACE_TYPE_BOOL    = 13,
} TraceType;

typedef struct TraceFieldStr { const char* data; size_t len; } TraceFieldStr;
typedef struct TraceFieldBytes { const uint8_t* data; size_t len; } TraceFieldBytes;
typedef struct TraceFieldNdarray {
    size_t ndim;
    const size_t* shape;
    const size_t* strides;
    const void* data;
    size_t elem_size;
    const char* descr;
} TraceFieldNdarray;

typedef struct TraceField {
    uint8_t type;
    const char* key;
    size_t key_len;
    union {
        int32_t          as_i32;
        int64_t          as_i64;
        uint32_t         as_u32;
        uint64_t         as_u64;
        float            as_f32;
        double           as_f64;
        bool             as_bool;
        TraceFieldStr    as_str;
        TraceFieldBytes  as_bytes;
        TraceFieldNdarray as_ndarray;
    } val;
} TraceField;

// Field macros
#ifdef __cplusplus
static inline TraceField _tf_i32(const char* k, int32_t v)  { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_I32;  f.key=k; f.key_len=__builtin_strlen(k); f.val.as_i32=v;  return f; }
static inline TraceField _tf_i64(const char* k, int64_t v)  { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_I64;  f.key=k; f.key_len=__builtin_strlen(k); f.val.as_i64=v;  return f; }
static inline TraceField _tf_u32(const char* k, uint32_t v) { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_U32;  f.key=k; f.key_len=__builtin_strlen(k); f.val.as_u32=v;  return f; }
static inline TraceField _tf_u64(const char* k, uint64_t v) { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_U64;  f.key=k; f.key_len=__builtin_strlen(k); f.val.as_u64=v;  return f; }
static inline TraceField _tf_f32(const char* k, float v)    { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_F32;  f.key=k; f.key_len=__builtin_strlen(k); f.val.as_f32=v;  return f; }
static inline TraceField _tf_f64(const char* k, double v)   { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_F64;  f.key=k; f.key_len=__builtin_strlen(k); f.val.as_f64=v;  return f; }
static inline TraceField _tf_bool(const char* k, bool v)    { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_BOOL; f.key=k; f.key_len=__builtin_strlen(k); f.val.as_bool=v; return f; }
static inline TraceField _tf_none(const char* k)            { TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_NONE; f.key=k; f.key_len=__builtin_strlen(k); return f; }
static inline TraceField _tf_str(const char* k, const char* s, size_t l) {
    TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_STR; f.key=k; f.key_len=__builtin_strlen(k); f.val.as_str.data=s; f.val.as_str.len=l; return f;
}
static inline TraceField _tf_bytes(const char* k, const uint8_t* d, size_t l) {
    TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_BYTES; f.key=k; f.key_len=__builtin_strlen(k); f.val.as_bytes.data=d; f.val.as_bytes.len=l; return f;
}
static inline TraceField _tf_ndarray(const char* k, size_t nd, const size_t* sh, const size_t* st, const void* d, size_t es, const char* de) {
    TraceField f; memset(&f,0,sizeof(f)); f.type=TRACE_TYPE_NDARRAY; f.key=k; f.key_len=__builtin_strlen(k);
    f.val.as_ndarray.ndim=nd; f.val.as_ndarray.shape=sh; f.val.as_ndarray.strides=st; f.val.as_ndarray.data=d; f.val.as_ndarray.elem_size=es; f.val.as_ndarray.descr=de;
    return f;
}
#define TI32(key, v)    _tf_i32(key, v)
#define TI64(key, v)    _tf_i64(key, v)
#define TU32(key, v)    _tf_u32(key, v)
#define TU64(key, v)    _tf_u64(key, v)
#define TF32(key, v)    _tf_f32(key, v)
#define TF64(key, v)    _tf_f64(key, v)
#define TBOOL(key, v)   _tf_bool(key, v)
#define TNONE(key)      _tf_none(key)
#define TSTR(key, s, l) _tf_str(key, s, l)
#define TBYTES(key, d, l) _tf_bytes(key, d, l)
#define TNDARRAY(key, nd, sh, st, d, es, de) _tf_ndarray(key, nd, sh, st, d, es, de)
#else
#define TI32(k, v)     { .type=TRACE_TYPE_I32,  .key=(k), .key_len=sizeof(k)-1, .val={.as_i32=(v)} }
#define TI64(k, v)     { .type=TRACE_TYPE_I64,  .key=(k), .key_len=sizeof(k)-1, .val={.as_i64=(v)} }
#define TU32(k, v)     { .type=TRACE_TYPE_U32,  .key=(k), .key_len=sizeof(k)-1, .val={.as_u32=(v)} }
#define TU64(k, v)     { .type=TRACE_TYPE_U64,  .key=(k), .key_len=sizeof(k)-1, .val={.as_u64=(v)} }
#define TF32(k, v)     { .type=TRACE_TYPE_F32,  .key=(k), .key_len=sizeof(k)-1, .val={.as_f32=(v)} }
#define TF64(k, v)     { .type=TRACE_TYPE_F64,  .key=(k), .key_len=sizeof(k)-1, .val={.as_f64=(v)} }
#define TBOOL(k, v)    { .type=TRACE_TYPE_BOOL, .key=(k), .key_len=sizeof(k)-1, .val={.as_bool=(v)} }
#define TNONE(k)       { .type=TRACE_TYPE_NONE, .key=(k), .key_len=sizeof(k)-1 }
#define TSTR(k, s, l)  { .type=TRACE_TYPE_STR,  .key=(k), .key_len=sizeof(k)-1, .val={.as_str={(s),(l)}} }
#define TBYTES(k, d, l){ .type=TRACE_TYPE_BYTES,.key=(k), .key_len=sizeof(k)-1, .val={.as_bytes={(d),(l)}} }
#define TNDARRAY(k, nd, sh, st, d, es, de) \
    { .type=TRACE_TYPE_NDARRAY, .key=(k), .key_len=sizeof(k)-1, \
      .val={.as_ndarray={ (nd),(sh),(st),(d),(es),(de) }} }
#endif

// Low-level size helpers
static inline size_t _trace_key_size(size_t key_len) {
    return 8 + align_up(key_len, 8);
}

static inline size_t trace_size_header(size_t id_len, size_t num_entries) {
    (void)num_entries;
    return 8 + id_len + 8;
}

static inline size_t trace_size_none(size_t key_len)  { return _trace_key_size(key_len) + 8; }
static inline size_t trace_size_i32(size_t key_len)   { return _trace_key_size(key_len) + 8 + 8; }
static inline size_t trace_size_i64(size_t key_len)   { return _trace_key_size(key_len) + 8 + 8; }
static inline size_t trace_size_u32(size_t key_len)   { return _trace_key_size(key_len) + 8 + 8; }
static inline size_t trace_size_u64(size_t key_len)   { return _trace_key_size(key_len) + 8 + 8; }
static inline size_t trace_size_f32(size_t key_len)   { return _trace_key_size(key_len) + 8 + 8; }
static inline size_t trace_size_f64(size_t key_len)   { return _trace_key_size(key_len) + 8 + 8; }
static inline size_t trace_size_bool(size_t key_len)  { return _trace_key_size(key_len) + 8 + 8; }
static inline size_t trace_size_str(size_t key_len, size_t str_len)   { return _trace_key_size(key_len) + 8 + 8 + align_up(str_len, 8); }
static inline size_t trace_size_bytes(size_t key_len, size_t data_len){ return _trace_key_size(key_len) + 8 + 8 + align_up(data_len, 8); }

// .npy header size: magic(6) + version(2) + header_len_field(2) + header_content (padded to 64B alignment)
static size_t _npy_header_size(size_t ndim, const char* descr) {
    // header dict: "{'descr': 'XX', 'fortran_order': False, 'shape': (N, M, ...), }\n"
    // We compute worst-case shape string: each dim up to 20 digits + ", "
    size_t dict_len = 10 + strlen(descr) + 3 + 25 + 10; // {'descr': '...',  'fortran_order': False, 'shape': (
    dict_len += ndim * 22; // each dim: up to 20 digits + ", "
    dict_len += 4; // ), }
    // Total preamble before dict content: magic(6) + version(2) + header_len(2) = 10
    size_t total = 10 + dict_len;
    // Pad to 64 byte alignment
    return align_up(total, 64);
}

static size_t _npy_total_data_size(size_t ndim, const size_t* shape, size_t elem_size) {
    size_t total = elem_size;
    for (size_t i = 0; i < ndim; i++) total *= shape[i];
    return total;
}

static inline size_t trace_size_ndarray(size_t key_len, size_t ndim, const size_t* shape, size_t elem_size, const char* descr) {
    if (ndim > TRACER_MAX_NDIM) return 0;
    size_t npy_size = _npy_header_size(ndim, descr) + _npy_total_data_size(ndim, shape, elem_size);
    return _trace_key_size(key_len) + 8 + 8 + align_up(npy_size, 8);
}

// Low-level write helpers
static inline uint8_t* trace_write_key(uint8_t* buf, const char* key, size_t key_len) {
    uint64_t kl = (uint64_t)key_len;
    memcpy(buf, &kl, 8); buf += 8;
    memcpy(buf, key, key_len);
    size_t padded = align_up(key_len, 8);
    if (padded > key_len) memset(buf + key_len, 0, padded - key_len);
    buf += padded;
    return buf;
}

static inline uint8_t* trace_write_header(uint8_t* buf, const char* id, size_t id_len, size_t num_entries) {
    uint64_t il = (uint64_t)id_len;
    memcpy(buf, &il, 8); buf += 8;
    memcpy(buf, id, id_len); buf += id_len;
    uint64_t ne = (uint64_t)num_entries;
    memcpy(buf, &ne, 8); buf += 8;
    return buf;
}

static inline uint8_t* _trace_write_type(uint8_t* buf, uint64_t type_code) {
    memcpy(buf, &type_code, 8);
    return buf + 8;
}

static inline uint8_t* trace_write_none(uint8_t* buf, const char* key, size_t key_len) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_NONE);
    return buf;
}

static inline uint8_t* trace_write_i64(uint8_t* buf, const char* key, size_t key_len, int64_t val) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_I64);
    memcpy(buf, &val, 8); buf += 8;
    return buf;
}

static inline uint8_t* trace_write_f64(uint8_t* buf, const char* key, size_t key_len, double val) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_F64);
    memcpy(buf, &val, 8); buf += 8;
    return buf;
}

static inline uint8_t* trace_write_i32(uint8_t* buf, const char* key, size_t key_len, int32_t val) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_I32);
    int64_t v64 = val;
    memcpy(buf, &v64, 8); buf += 8;
    return buf;
}

static inline uint8_t* trace_write_u32(uint8_t* buf, const char* key, size_t key_len, uint32_t val) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_U32);
    uint64_t v64 = val;
    memcpy(buf, &v64, 8); buf += 8;
    return buf;
}

static inline uint8_t* trace_write_u64(uint8_t* buf, const char* key, size_t key_len, uint64_t val) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_U64);
    memcpy(buf, &val, 8); buf += 8;
    return buf;
}

static inline uint8_t* trace_write_f32(uint8_t* buf, const char* key, size_t key_len, float val) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_F32);
    double v64 = val;
    memcpy(buf, &v64, 8); buf += 8;
    return buf;
}

static inline uint8_t* trace_write_bool(uint8_t* buf, const char* key, size_t key_len, bool val) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_BOOL);
    int64_t v64 = val ? 1 : 0;
    memcpy(buf, &v64, 8); buf += 8;
    return buf;
}

static inline uint8_t* trace_write_str(uint8_t* buf, const char* key, size_t key_len, const char* str, size_t str_len) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_STR);
    uint64_t sl = (uint64_t)str_len;
    memcpy(buf, &sl, 8); buf += 8;
    memcpy(buf, str, str_len);
    size_t padded = align_up(str_len, 8);
    if (padded > str_len) memset(buf + str_len, 0, padded - str_len);
    buf += padded;
    return buf;
}

static inline uint8_t* trace_write_bytes(uint8_t* buf, const char* key, size_t key_len, const uint8_t* data, size_t data_len) {
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_BYTES);
    uint64_t dl = (uint64_t)data_len;
    memcpy(buf, &dl, 8); buf += 8;
    memcpy(buf, data, data_len);
    size_t padded = align_up(data_len, 8);
    if (padded > data_len) memset(buf + data_len, 0, padded - data_len);
    buf += padded;
    return buf;
}

// Write .npy header and copy strided data into contiguous layout
static uint8_t* trace_write_ndarray(uint8_t* buf, const char* key, size_t key_len,
                                     size_t ndim, const size_t* shape, const size_t* strides,
                                     const void* data, size_t elem_size, const char* descr)
{
    buf = trace_write_key(buf, key, key_len);
    buf = _trace_write_type(buf, TRACE_TYPE_NDARRAY);

    size_t npy_hdr_size = _npy_header_size(ndim, descr);
    size_t data_size = _npy_total_data_size(ndim, shape, elem_size);
    size_t npy_total = npy_hdr_size + data_size;
    uint64_t npy_len = (uint64_t)npy_total;
    memcpy(buf, &npy_len, 8); buf += 8;

    uint8_t* npy_start = buf;

    // .npy magic and version
    buf[0] = 0x93; buf[1] = 'N'; buf[2] = 'U'; buf[3] = 'M'; buf[4] = 'P'; buf[5] = 'Y';
    buf[6] = 1; buf[7] = 0; // version 1.0
    buf += 8;

    // Build header dict
    uint8_t* dict_start = buf + 2; // skip header_len field
    int dict_len = 0;

    // Shape string
    char shape_buf[512];
    int spos = 0;
    shape_buf[spos++] = '(';
    for (size_t i = 0; i < ndim; i++) {
        if (i > 0) { shape_buf[spos++] = ','; shape_buf[spos++] = ' '; }
        spos += snprintf(shape_buf + spos, sizeof(shape_buf) - (size_t)spos, "%zu", shape[i]);
    }
    if (ndim == 1) shape_buf[spos++] = ',';
    shape_buf[spos++] = ')';
    shape_buf[spos] = '\0';

    dict_len = snprintf((char*)dict_start, npy_hdr_size - 10,
        "{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shape_buf);

    // Pad with spaces to align total header to 64 bytes, end with \n
    size_t used = 10 + (size_t)dict_len;
    size_t padded_total = align_up(used + 1, 64); // +1 for the \n
    size_t pad_count = padded_total - used - 1;
    memset(dict_start + dict_len, ' ', pad_count);
    dict_start[dict_len + pad_count] = '\n';

    // Write header_len (2 bytes LE) = padded_total - 10
    uint16_t header_len_val = (uint16_t)(padded_total - 10);
    memcpy(buf, &header_len_val, 2);

    buf = npy_start + padded_total;

    // Copy data using strides (element by element for non-contiguous)
    if (ndim == 0) {
        memcpy(buf, data, elem_size);
        buf += elem_size;
    } else {
        // Recursive strided copy using iterative approach with index array
        size_t total_elems = 1;
        for (size_t i = 0; i < ndim; i++) total_elems *= shape[i];

        size_t indices[TRACER_MAX_NDIM];
        if (ndim > TRACER_MAX_NDIM) return npy_start; // graceful fail: return without writing data
        memset(indices, 0, ndim * sizeof(size_t));

        for (size_t e = 0; e < total_elems; e++) {
            // Compute source offset from indices and strides
            size_t src_offset = 0;
            for (size_t d = 0; d < ndim; d++) src_offset += indices[d] * strides[d];
            memcpy(buf, (const uint8_t*)data + src_offset, elem_size);
            buf += elem_size;

            // Increment indices (last dimension first, C-order)
            for (size_t d = ndim; d > 0; d--) {
                indices[d-1]++;
                if (indices[d-1] < shape[d-1]) break;
                indices[d-1] = 0;
            }
        }
    }

    // Pad npy data to 8-byte alignment
    size_t npy_padded = align_up(npy_total, 8);
    if (npy_padded > npy_total) {
        memset(npy_start + npy_total, 0, npy_padded - npy_total);
    }
    return npy_start + npy_padded;
}

// _trace_field_size / _trace_field_write dispatch on TraceField.type
static inline size_t _trace_field_size(const TraceField* f) {
    switch (f->type) {
        case TRACE_TYPE_NONE:    return trace_size_none(f->key_len);
        case TRACE_TYPE_I32:     return trace_size_i32(f->key_len);
        case TRACE_TYPE_I64:     return trace_size_i64(f->key_len);
        case TRACE_TYPE_U32:     return trace_size_u32(f->key_len);
        case TRACE_TYPE_U64:     return trace_size_u64(f->key_len);
        case TRACE_TYPE_F32:     return trace_size_f32(f->key_len);
        case TRACE_TYPE_F64:     return trace_size_f64(f->key_len);
        case TRACE_TYPE_BOOL:    return trace_size_bool(f->key_len);
        case TRACE_TYPE_STR:     return trace_size_str(f->key_len, f->val.as_str.len);
        case TRACE_TYPE_BYTES:   return trace_size_bytes(f->key_len, f->val.as_bytes.len);
        case TRACE_TYPE_NDARRAY: return trace_size_ndarray(f->key_len, f->val.as_ndarray.ndim, f->val.as_ndarray.shape, f->val.as_ndarray.elem_size, f->val.as_ndarray.descr);
        default: ASSERT(0 && "unknown trace type"); return 0;
    }
}

static inline uint8_t* _trace_field_write(uint8_t* buf, const TraceField* f) {
    switch (f->type) {
        case TRACE_TYPE_NONE:    return trace_write_none(buf, f->key, f->key_len);
        case TRACE_TYPE_I32:     return trace_write_i32(buf, f->key, f->key_len, f->val.as_i32);
        case TRACE_TYPE_I64:     return trace_write_i64(buf, f->key, f->key_len, f->val.as_i64);
        case TRACE_TYPE_U32:     return trace_write_u32(buf, f->key, f->key_len, f->val.as_u32);
        case TRACE_TYPE_U64:     return trace_write_u64(buf, f->key, f->key_len, f->val.as_u64);
        case TRACE_TYPE_F32:     return trace_write_f32(buf, f->key, f->key_len, f->val.as_f32);
        case TRACE_TYPE_F64:     return trace_write_f64(buf, f->key, f->key_len, f->val.as_f64);
        case TRACE_TYPE_BOOL:    return trace_write_bool(buf, f->key, f->key_len, f->val.as_bool);
        case TRACE_TYPE_STR:     return trace_write_str(buf, f->key, f->key_len, f->val.as_str.data, f->val.as_str.len);
        case TRACE_TYPE_BYTES:   return trace_write_bytes(buf, f->key, f->key_len, f->val.as_bytes.data, f->val.as_bytes.len);
        case TRACE_TYPE_NDARRAY: return trace_write_ndarray(buf, f->key, f->key_len, f->val.as_ndarray.ndim, f->val.as_ndarray.shape, f->val.as_ndarray.strides, f->val.as_ndarray.data, f->val.as_ndarray.elem_size, f->val.as_ndarray.descr);
        default: ASSERT(0 && "unknown trace type"); return buf;
    }
}

// ---------------------------------------------------------------------------
// Tracepoint registry and hash table
// ---------------------------------------------------------------------------

typedef struct Tracepoint {
    const char* name;
    size_t      name_len;
    _Atomic(uint32_t) subscriber_mask;
} Tracepoint;

typedef struct TracepointRegistry {
    Tracepoint tracepoints[TRACER_MAX_TRACEPOINTS];
    _Atomic(uint32_t) count;
    bool frozen;
} TracepointRegistry;

typedef struct TracepointHashEntry {
    const char* key;
    size_t      key_len;
    Tracepoint* tracepoint;
} TracepointHashEntry;

typedef struct TracepointHashTable {
    TracepointHashEntry entries[TRACER_HASH_TABLE_CAPACITY];
} TracepointHashTable;

// ---------------------------------------------------------------------------
// Subscribers, TLV protocol, and global tracer (defined here so that
// tracepoint_register can access g_tracer before the function bodies)
// ---------------------------------------------------------------------------

typedef enum SubscriberType { SUBSCRIBER_NONE = 0, SUBSCRIBER_TCP, SUBSCRIBER_SQLITE } SubscriberType;

typedef struct Subscriber {
    SubscriberType type;
    MpscRingBuffer ring_buffer;
    Thread         consumer_thread;
    _Atomic(bool)  active;
    _Atomic(bool)  connected;
    uint32_t       index;
    union {
        struct { char host[256]; uint16_t port; socket_t sock; } tcp;
#ifdef TRACER_SQLITE_ENABLED
        struct { void* db; char path[512]; } sqlite;
#endif
    } cfg;
} Subscriber;

#define TLV_HEADER_SIZE          16
#define MSG_TRACE_EVENT          0x200001
#define MSG_LIST_TRACEPOINTS     0x200002
#define MSG_LIST_TP_RESPONSE     0x200003
#define MSG_ENABLE_PATTERN       0x200004
#define MSG_ENABLE_PATTERN_RESP  0x200005
#define MSG_DISABLE_PATTERN      0x200006
#define MSG_DISABLE_PATTERN_RESP 0x200007

typedef struct Tracer {
    TracepointRegistry  registry;
    TracepointHashTable hash_table;
    Subscriber          subscribers[TRACER_MAX_SUBSCRIBERS];
    _Atomic(bool)       initialized;
} Tracer;

static Tracer g_tracer;

static inline bool tracepoint_enabled(const Tracepoint* tp) {
    return atomic_load_explicit(&((Tracepoint*)tp)->subscriber_mask, memory_order_relaxed) != 0;
}

static Tracepoint* tracepoint_register(const char* name) {
    uint32_t idx = atomic_fetch_add_explicit(&g_tracer.registry.count, 1, memory_order_relaxed);
    ASSERT(idx < TRACER_MAX_TRACEPOINTS && "too many tracepoints");
    ASSERT(!g_tracer.registry.frozen && "cannot register tracepoints after tracer_init()");
    Tracepoint* tp = &g_tracer.registry.tracepoints[idx];
    tp->name = name;
    tp->name_len = strlen(name);
    atomic_store_explicit(&tp->subscriber_mask, 0, memory_order_relaxed);
    return tp;
}

#ifdef _MSC_VER
#define TRACEPOINT_DEFINE(varname, namestr) \
    static Tracepoint* varname; \
    static void __cdecl _tp_init_##varname(void) { varname = tracepoint_register(namestr); } \
    __pragma(section(".CRT$XCU", read)) \
    __declspec(allocate(".CRT$XCU")) static void(__cdecl* _tp_ptr_##varname)(void) = _tp_init_##varname;
#elif defined(__cplusplus)
#define TRACEPOINT_DEFINE(varname, namestr) \
    static Tracepoint* varname; \
    namespace { struct _tp_reg_##varname { _tp_reg_##varname() { varname = tracepoint_register(namestr); } } _tp_inst_##varname; }
#else
#define TRACEPOINT_DEFINE(varname, namestr) \
    static Tracepoint* varname; \
    __attribute__((constructor)) static void _tp_init_##varname(void) { varname = tracepoint_register(namestr); }
#endif

// Hash table (FNV-1a, open addressing, linear probing)
static inline uint32_t _hash_fnv1a(const char* key, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)key[i];
        h *= 16777619u;
    }
    return h;
}

static void _tracepoint_ht_init(TracepointHashTable* ht) {
    memset(ht->entries, 0, sizeof(ht->entries));
}

static void _tracepoint_ht_insert(TracepointHashTable* ht, Tracepoint* tp) {
    uint32_t mask = TRACER_HASH_TABLE_CAPACITY - 1;
    uint32_t idx = _hash_fnv1a(tp->name, tp->name_len) & mask;
    while (ht->entries[idx].key != NULL) {
        idx = (idx + 1) & mask;
    }
    ht->entries[idx].key = tp->name;
    ht->entries[idx].key_len = tp->name_len;
    ht->entries[idx].tracepoint = tp;
}

static Tracepoint* _tracepoint_ht_find(const TracepointHashTable* ht, const char* name, size_t name_len) {
    uint32_t mask = TRACER_HASH_TABLE_CAPACITY - 1;
    uint32_t idx = _hash_fnv1a(name, name_len) & mask;
    while (ht->entries[idx].key != NULL) {
        if (ht->entries[idx].key_len == name_len && memcmp(ht->entries[idx].key, name, name_len) == 0) {
            return ht->entries[idx].tracepoint;
        }
        idx = (idx + 1) & mask;
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// Sleep helper
// ---------------------------------------------------------------------------

static void _tracer_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
#endif
}

// ---------------------------------------------------------------------------
// TCP subscriber consumer thread
// ---------------------------------------------------------------------------

static bool _tcp_send_all(socket_t sock, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    while (len > 0) {
#ifdef _WIN32
        int sent = send(sock, (const char*)p, (int)len, 0);
#else
        ssize_t sent = send(sock, p, len, 0);
#endif
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (sent == 0) return false;
        p += sent;
        len -= (size_t)sent;
    }
    return true;
}

static bool _tcp_send_handshake(socket_t sock) {
    uint8_t hs[14];
    memcpy(hs, "AMBR", 4);
    uint32_t name_len = 6;
    memcpy(hs + 4, &name_len, 4);
    memcpy(hs + 8, "tracer", 6);
    return _tcp_send_all(sock, hs, sizeof(hs));
}

static bool _tcp_sendv(socket_t sock, const void* a, size_t a_len, const void* b, size_t b_len) {
#ifdef _WIN32
    WSABUF bufs[2];
    DWORD count = 0;
    bufs[0].buf = (char*)a; bufs[0].len = (ULONG)a_len;
    bufs[1].buf = (char*)b; bufs[1].len = (ULONG)b_len;
    int nbufs = b_len > 0 ? 2 : 1;
    size_t total = a_len + b_len;
    while (total > 0) {
        DWORD sent = 0;
        if (WSASend(sock, bufs, nbufs, &sent, 0, NULL, NULL) == SOCKET_ERROR) return false;
        if (sent == 0) return false;
        total -= sent;
        // Advance buffers
        for (int i = 0; i < nbufs && sent > 0; i++) {
            if (sent >= bufs[i].len) { sent -= bufs[i].len; bufs[i].len = 0; bufs[i].buf += bufs[i].len; }
            else { bufs[i].buf += sent; bufs[i].len -= (ULONG)sent; sent = 0; }
        }
    }
    return true;
#else
    struct iovec iov[2];
    iov[0].iov_base = (void*)a; iov[0].iov_len = a_len;
    iov[1].iov_base = (void*)b; iov[1].iov_len = b_len;
    int nbufs = b_len > 0 ? 2 : 1;
    size_t total = a_len + b_len;
    while (total > 0) {
        ssize_t sent = writev(sock, iov, nbufs);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (sent == 0) return false;
        total -= (size_t)sent;
        // Advance buffers
        for (int i = 0; i < nbufs && sent > 0; i++) {
            if ((size_t)sent >= iov[i].iov_len) { sent -= (ssize_t)iov[i].iov_len; iov[i].iov_len = 0; }
            else { iov[i].iov_base = (uint8_t*)iov[i].iov_base + sent; iov[i].iov_len -= (size_t)sent; sent = 0; }
        }
    }
    return true;
#endif
}

static bool _tcp_send_tlv(socket_t sock, uint32_t msg_type, const uint8_t* payload, size_t payload_len) {
    uint8_t header[TLV_HEADER_SIZE];
    uint32_t format = 0;
    uint64_t length = (uint64_t)payload_len;
    memcpy(header + 0, &msg_type, 4);
    memcpy(header + 4, &format, 4);
    memcpy(header + 8, &length, 8);
    return _tcp_sendv(sock, header, TLV_HEADER_SIZE, payload, payload_len);
}

static THREAD_PROC(_tcp_consumer_thread) {
    Subscriber* sub = (Subscriber*)data;

    while (atomic_load_explicit(&sub->active, memory_order_relaxed)) {
        // Connect phase
        if (!atomic_load_explicit(&sub->connected, memory_order_relaxed)) {
            socket_t sock = 0;
            Result res = socket_connect_blocking(sub->cfg.tcp.host, sub->cfg.tcp.port, &sock);
            if (res != SUCCESS) {
                _tracer_sleep_ms(TRACER_TCP_RECONNECT_MS);
                continue;
            }
            sub->cfg.tcp.sock = sock;
            if (!_tcp_send_handshake(sock)) {
                socket_close(sock);
                continue;
            }
            atomic_store_explicit(&sub->connected, true, memory_order_release);
        }

        // Drain ring buffer
        uint8_t* payload;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(&sub->ring_buffer, &payload);

        if (!atomic_load_explicit(&sub->active, memory_order_relaxed)) {
            mpsc_ring_buffer_lock_release_read(&sub->ring_buffer, sz);
            break;
        }

        if (!_tcp_send_tlv(sub->cfg.tcp.sock, MSG_TRACE_EVENT, payload, sz)) {
            atomic_store_explicit(&sub->connected, false, memory_order_relaxed);
            socket_close(sub->cfg.tcp.sock);
        }

        mpsc_ring_buffer_lock_release_read(&sub->ring_buffer, sz);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// SQLite subscriber consumer thread
// ---------------------------------------------------------------------------

#ifdef TRACER_SQLITE_ENABLED
#include <sqlite3.h>

#define SQLITE_MAX_TABLES 64
#define SQLITE_MAX_COLUMNS 64

typedef struct SqliteTableInfo {
    char name[256];
    size_t name_len;
    sqlite3_stmt* insert_stmt;
} SqliteTableInfo;

typedef struct SqliteState {
    SqliteTableInfo tables[SQLITE_MAX_TABLES];
    uint32_t        table_count;
} SqliteState;

static uint64_t _read_u64(const uint8_t* p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

// Deserialize binary payload and insert into SQLite
static void _sqlite_process_entry(sqlite3* db, SqliteState* state, const uint8_t* payload, size_t payload_len) {
    const uint8_t* p = payload;
    const uint8_t* end = payload + payload_len;

    // Read identifier
    uint64_t id_len = _read_u64(p); p += 8;
    const char* identifier = (const char*)p; p += id_len;

    // Read num_entries
    uint64_t num_entries = _read_u64(p); p += 8;

    // Find existing table
    SqliteTableInfo* table = NULL;
    for (uint32_t i = 0; i < state->table_count; i++) {
        if (state->tables[i].name_len == id_len && memcmp(state->tables[i].name, identifier, id_len) == 0) {
            table = &state->tables[i];
            break;
        }
    }

    // Read all keys and values into temp arrays
    const char* keys[SQLITE_MAX_COLUMNS];
    size_t key_lens[SQLITE_MAX_COLUMNS];
    uint64_t types[SQLITE_MAX_COLUMNS];
    const uint8_t* value_ptrs[SQLITE_MAX_COLUMNS];

    ASSERT(num_entries <= SQLITE_MAX_COLUMNS);

    for (uint64_t i = 0; i < num_entries; i++) {
        if (p >= end) return;
        uint64_t kl = _read_u64(p); p += 8;
        keys[i] = (const char*)p;
        key_lens[i] = (size_t)kl;
        p += align_up(kl, 8);

        types[i] = _read_u64(p); p += 8;
        value_ptrs[i] = p;

        switch (types[i]) {
            case TRACE_TYPE_NONE: break;
            case TRACE_TYPE_I32: case TRACE_TYPE_I64: case TRACE_TYPE_U32:
            case TRACE_TYPE_U64: case TRACE_TYPE_F32: case TRACE_TYPE_F64:
            case TRACE_TYPE_BOOL: p += 8; break;
            case TRACE_TYPE_STR: case TRACE_TYPE_BYTES: case TRACE_TYPE_NDARRAY: {
                uint64_t dl = _read_u64(p); p += 8;
                p += align_up(dl, 8);
            } break;
            default: return; // unknown type, skip entry
        }
    }

    // Create table if new
    if (!table) {
        if (state->table_count >= SQLITE_MAX_TABLES) return;
        table = &state->tables[state->table_count++];
        memcpy(table->name, identifier, id_len);
        table->name[id_len] = '\0';
        table->name_len = id_len;
        table->column_count = (uint32_t)num_entries;
        table->insert_stmt = NULL;

        // Build CREATE TABLE
        char sql[4096];
        int pos = snprintf(sql, sizeof(sql), "CREATE TABLE IF NOT EXISTS \"%.*s\" (", (int)id_len, identifier);
        for (uint64_t i = 0; i < num_entries; i++) {
            if (i > 0) { sql[pos++] = ','; sql[pos++] = ' '; }
            pos += snprintf(sql + pos, sizeof(sql) - (size_t)pos, "\"%.*s\"", (int)key_lens[i], keys[i]);
            memcpy(table->columns[i], keys[i], key_lens[i]);
            table->columns[i][key_lens[i]] = '\0';

            switch (types[i]) {
                case TRACE_TYPE_I32: case TRACE_TYPE_I64: case TRACE_TYPE_U32:
                case TRACE_TYPE_U64: case TRACE_TYPE_BOOL:
                    pos += snprintf(sql + pos, sizeof(sql) - (size_t)pos, " INTEGER"); break;
                case TRACE_TYPE_F32: case TRACE_TYPE_F64:
                    pos += snprintf(sql + pos, sizeof(sql) - (size_t)pos, " REAL"); break;
                case TRACE_TYPE_STR:
                    pos += snprintf(sql + pos, sizeof(sql) - (size_t)pos, " TEXT"); break;
                default:
                    pos += snprintf(sql + pos, sizeof(sql) - (size_t)pos, " BLOB"); break;
            }
        }
        pos += snprintf(sql + pos, sizeof(sql) - (size_t)pos, ")");
        sqlite3_exec(db, sql, NULL, NULL, NULL);

        // Prepare INSERT statement
        pos = snprintf(sql, sizeof(sql), "INSERT INTO \"%.*s\" VALUES (", (int)id_len, identifier);
        for (uint64_t i = 0; i < num_entries; i++) {
            if (i > 0) sql[pos++] = ',';
            sql[pos++] = '?';
        }
        sql[pos++] = ')';
        sql[pos] = '\0';
        sqlite3_prepare_v2(db, sql, pos, &table->insert_stmt, NULL);
    } else {
        // Verify schema matches
        ASSERT(table->column_count == (uint32_t)num_entries && "tracepoint schema mismatch");
        for (uint64_t i = 0; i < num_entries; i++) {
            ASSERT(key_lens[i] == strlen(table->columns[i]) &&
                   memcmp(keys[i], table->columns[i], key_lens[i]) == 0 &&
                   "tracepoint column mismatch");
        }
    }

    // Bind values and execute
    sqlite3_reset(table->insert_stmt);
    for (uint64_t i = 0; i < num_entries; i++) {
        int col = (int)(i + 1);
        const uint8_t* vp = value_ptrs[i];
        switch (types[i]) {
            case TRACE_TYPE_NONE: sqlite3_bind_null(table->insert_stmt, col); break;
            case TRACE_TYPE_I32: case TRACE_TYPE_I64: case TRACE_TYPE_BOOL: {
                int64_t v; memcpy(&v, vp, 8);
                sqlite3_bind_int64(table->insert_stmt, col, v);
            } break;
            case TRACE_TYPE_U32: case TRACE_TYPE_U64: {
                int64_t v; memcpy(&v, vp, 8);
                sqlite3_bind_int64(table->insert_stmt, col, v);
            } break;
            case TRACE_TYPE_F32: case TRACE_TYPE_F64: {
                double v; memcpy(&v, vp, 8);
                sqlite3_bind_double(table->insert_stmt, col, v);
            } break;
            case TRACE_TYPE_STR: {
                uint64_t sl = _read_u64(vp);
                sqlite3_bind_text(table->insert_stmt, col, (const char*)(vp + 8), (int)sl, SQLITE_TRANSIENT);
            } break;
            case TRACE_TYPE_BYTES: case TRACE_TYPE_NDARRAY: {
                uint64_t dl = _read_u64(vp);
                sqlite3_bind_blob(table->insert_stmt, col, vp + 8, (int)dl, SQLITE_TRANSIENT);
            } break;
            default: break;
        }
    }
    sqlite3_step(table->insert_stmt);
}

static THREAD_PROC(_sqlite_consumer_thread) {
    Subscriber* sub = (Subscriber*)data;
    sqlite3* db = (sqlite3*)sub->cfg.sqlite.db;

    sqlite3_exec(db, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);

    SqliteState* state = (SqliteState*)calloc(1, sizeof(SqliteState));
    ASSERT(state && "failed to allocate sqlite state");
    uint32_t commit_counter = 0;

    while (atomic_load_explicit(&sub->active, memory_order_relaxed)) {
        uint8_t* payload;
        size_t sz = mpsc_ring_buffer_lock_acquire_read(&sub->ring_buffer, &payload);

        if (!atomic_load_explicit(&sub->active, memory_order_relaxed)) {
            mpsc_ring_buffer_lock_release_read(&sub->ring_buffer, sz);
            break;
        }

        _sqlite_process_entry(db, state, payload, sz);
        mpsc_ring_buffer_lock_release_read(&sub->ring_buffer, sz);

        if (++commit_counter >= 64) {
            sqlite3_exec(db, "COMMIT; BEGIN", NULL, NULL, NULL);
            commit_counter = 0;
        }
    }

    // Final commit and cleanup
    sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
    for (uint32_t i = 0; i < state->table_count; i++) {
        if (state->tables[i].insert_stmt) sqlite3_finalize(state->tables[i].insert_stmt);
    }
    free(state);

    return 0;
}
#endif // TRACER_SQLITE_ENABLED

// ---------------------------------------------------------------------------
// Subscriber management
// ---------------------------------------------------------------------------

static int tracer_add_tcp_subscriber(const char* host, uint16_t port) {
    for (int i = 0; i < TRACER_MAX_SUBSCRIBERS; i++) {
        if (g_tracer.subscribers[i].type == SUBSCRIBER_NONE) {
            Subscriber* sub = &g_tracer.subscribers[i];
            sub->type = SUBSCRIBER_TCP;
            sub->index = (uint32_t)i;
            atomic_store_explicit(&sub->active, true, memory_order_relaxed);
            atomic_store_explicit(&sub->connected, false, memory_order_relaxed);
            snprintf(sub->cfg.tcp.host, sizeof(sub->cfg.tcp.host), "%s", host);
            sub->cfg.tcp.port = port;
            mpsc_ring_buffer_create(&sub->ring_buffer, TRACER_SUBSCRIBER_BUFFER_SIZE);
            create_thread(_tcp_consumer_thread, sub, &sub->consumer_thread);
            return i;
        }
    }
    return -1;
}

#ifdef TRACER_SQLITE_ENABLED
static int tracer_add_sqlite_subscriber(const char* db_path) {
    for (int i = 0; i < TRACER_MAX_SUBSCRIBERS; i++) {
        if (g_tracer.subscribers[i].type == SUBSCRIBER_NONE) {
            Subscriber* sub = &g_tracer.subscribers[i];
            sub->type = SUBSCRIBER_SQLITE;
            sub->index = (uint32_t)i;
            atomic_store_explicit(&sub->active, true, memory_order_relaxed);
            snprintf(sub->cfg.sqlite.path, sizeof(sub->cfg.sqlite.path), "%s", db_path);

            sqlite3* db;
            int rc = sqlite3_open(db_path, &db);
            ASSERT(rc == SQLITE_OK && "failed to open sqlite database");
            sub->cfg.sqlite.db = db;

            sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);

            mpsc_ring_buffer_create(&sub->ring_buffer, TRACER_SUBSCRIBER_BUFFER_SIZE);
            create_thread(_sqlite_consumer_thread, sub, &sub->consumer_thread);
            return i;
        }
    }
    return -1;
}
#endif

static bool tracer_subscribe(int subscriber_idx, const char* tracepoint_name) {
    Tracepoint* tp = _tracepoint_ht_find(&g_tracer.hash_table, tracepoint_name, strlen(tracepoint_name));
    if (!tp) return false;
    uint32_t bit = 1u << (uint32_t)subscriber_idx;
    atomic_fetch_or_explicit(&tp->subscriber_mask, bit, memory_order_relaxed);
    return true;
}

static bool tracer_unsubscribe(int subscriber_idx, const char* tracepoint_name) {
    Tracepoint* tp = _tracepoint_ht_find(&g_tracer.hash_table, tracepoint_name, strlen(tracepoint_name));
    if (!tp) return false;
    uint32_t bit = 1u << (uint32_t)subscriber_idx;
    atomic_fetch_and_explicit(&tp->subscriber_mask, ~bit, memory_order_relaxed);
    return true;
}

static void tracer_subscribe_all(int subscriber_idx) {
    uint32_t count = atomic_load_explicit(&g_tracer.registry.count, memory_order_relaxed);
    uint32_t bit = 1u << (uint32_t)subscriber_idx;
    for (uint32_t i = 0; i < count; i++) {
        atomic_fetch_or_explicit(&g_tracer.registry.tracepoints[i].subscriber_mask, bit, memory_order_relaxed);
    }
}

static void tracer_unsubscribe_all(int subscriber_idx) {
    uint32_t count = atomic_load_explicit(&g_tracer.registry.count, memory_order_relaxed);
    uint32_t bit = 1u << (uint32_t)subscriber_idx;
    for (uint32_t i = 0; i < count; i++) {
        atomic_fetch_and_explicit(&g_tracer.registry.tracepoints[i].subscriber_mask, ~bit, memory_order_relaxed);
    }
}

static void tracer_remove_subscriber(int idx) {
    Subscriber* sub = &g_tracer.subscribers[idx];
    if (sub->type == SUBSCRIBER_NONE) return;

    // Unsubscribe from all tracepoints
    tracer_unsubscribe_all(idx);

    // Signal consumer thread to stop
    atomic_store_explicit(&sub->active, false, memory_order_release);

    // Write a dummy entry to wake the consumer if it's blocked
    uint8_t* dummy = mpsc_ring_buffer_try_reserve_write(&sub->ring_buffer, 1);
    if (dummy) {
        dummy[0] = 0;
        mpsc_ring_buffer_commit_write(&sub->ring_buffer, dummy);
    }

    join_thread(&sub->consumer_thread);

    if (sub->type == SUBSCRIBER_TCP) {
        if (atomic_load_explicit(&sub->connected, memory_order_relaxed)) {
            socket_close(sub->cfg.tcp.sock);
        }
    }
#ifdef TRACER_SQLITE_ENABLED
    if (sub->type == SUBSCRIBER_SQLITE) {
        sqlite3_close((sqlite3*)sub->cfg.sqlite.db);
    }
#endif

    mpsc_ring_buffer_destroy(&sub->ring_buffer);
    sub->type = SUBSCRIBER_NONE;
}

// ---------------------------------------------------------------------------
// Tracer lifecycle
// ---------------------------------------------------------------------------

static void tracer_init(void) {
    ASSERT(!atomic_load_explicit(&g_tracer.initialized, memory_order_relaxed));

    g_tracer.registry.frozen = true;

    _tracepoint_ht_init(&g_tracer.hash_table);
    uint32_t count = atomic_load_explicit(&g_tracer.registry.count, memory_order_relaxed);
    for (uint32_t i = 0; i < count; i++) {
        _tracepoint_ht_insert(&g_tracer.hash_table, &g_tracer.registry.tracepoints[i]);
    }

    memset(g_tracer.subscribers, 0, sizeof(g_tracer.subscribers));
    atomic_store_explicit(&g_tracer.initialized, true, memory_order_release);
}

static void tracer_close(void) {
    for (int i = 0; i < TRACER_MAX_SUBSCRIBERS; i++) {
        tracer_remove_subscriber(i);
    }
    atomic_store_explicit(&g_tracer.initialized, false, memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// TRACE macro
// ---------------------------------------------------------------------------
#ifdef _MSC_VER
#include <intrin.h>
static inline int _trace_ctz_msvc(uint32_t x) { unsigned long idx; _BitScanForward(&idx, x); return (int)idx; }
#define _TRACE_CTZ(x) _trace_ctz_msvc(x)
#else
#define _TRACE_CTZ(x) __builtin_ctz(x)
#endif

static inline void _trace_emit(Tracepoint* tp, const TraceField* fields, size_t nfields) {
    uint32_t mask = atomic_load_explicit(&tp->subscriber_mask, memory_order_relaxed);

    // Compute total size once (same for all subscribers)
    size_t sz = trace_size_header(tp->name_len, nfields);
    for (size_t i = 0; i < nfields; i++)
        sz += _trace_field_size(&fields[i]);

    while (mask) {
        int idx = _TRACE_CTZ(mask);
        mask &= mask - 1;
        Subscriber* sub = &g_tracer.subscribers[idx];

        uint8_t* buf;
#if TRACER_QUEUE_FULL_POLICY
        buf = mpsc_ring_buffer_wait_reserve_write(&sub->ring_buffer, sz);
#else
        buf = mpsc_ring_buffer_try_reserve_write(&sub->ring_buffer, sz);
#endif
        if (!buf) continue;

        uint8_t* p = trace_write_header(buf, tp->name, tp->name_len, nfields);
        for (size_t i = 0; i < nfields; i++)
            p = _trace_field_write(p, &fields[i]);

        mpsc_ring_buffer_commit_write(&sub->ring_buffer, buf);
    }
}

#define TRACE(tp, ...) \
    do { \
        if (tracepoint_enabled(tp)) { \
            TraceField _trace_fields[] = { __VA_ARGS__ }; \
            _trace_emit((tp), _trace_fields, sizeof(_trace_fields)/sizeof(_trace_fields[0])); \
        } \
    } while (0)

// Low-level manual API
#define TRACE_BEGIN(tp) \
    do { \
        uint32_t _trace_mask = atomic_load_explicit(&(tp)->subscriber_mask, memory_order_relaxed); \
        if (_trace_mask) { \
            while (_trace_mask) { \
                int _trace_sub_idx = _TRACE_CTZ(_trace_mask); \
                _trace_mask &= _trace_mask - 1; \
                Subscriber* _trace_sub = &g_tracer.subscribers[_trace_sub_idx]; \
                SubscriberType _trace_sub_type = _trace_sub->type; \
                (void)_trace_sub_type;

#define TRACE_RESERVE(sz) \
                uint8_t* _trace_buf; \
                if (TRACER_QUEUE_FULL_POLICY) \
                    _trace_buf = mpsc_ring_buffer_wait_reserve_write(&_trace_sub->ring_buffer, (sz)); \
                else \
                    _trace_buf = mpsc_ring_buffer_try_reserve_write(&_trace_sub->ring_buffer, (sz)); \
                if (_trace_buf) {

#define TRACE_END() \
                    mpsc_ring_buffer_commit_write(&_trace_sub->ring_buffer, _trace_buf); \
                } \
            } \
        } \
    } while (0)

#endif

#endif // TRACING_H

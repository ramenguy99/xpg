#include <stdatomic.h>
#include <stdint.h>
#include <stdbool.h>
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

// Socket utils
#ifdef _WIN32
typedef SOCKET socket_t;
#else
typedef int socket_t;
#endif

#define CACHE_LINE_SIZE 64

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

Result socket_init();
Result socket_listen(uint16_t port, int backlog, bool only_ipv4, bool only_localhost, socket_t* socket);
Result socket_accept(socket_t listening_socket, int timeout, socket_t* socket);
Result socket_connect( const char* addr, uint16_t port, TcpConnection* connection_socket);
void socket_close(socket_t socket);
Result socket_connect_blocking( const char* addr, uint16_t port, socket_t* connection_socket);
void socket_close_connection(TcpConnection* connection);

#if 1
// #ifdef TRACING_IMPLEMENTATION

Result
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

Result
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

Result
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

void
socket_close(socket_t socket)
{
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

Result
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

void
socket_close_connection(TcpConnection* connection) {
    socket_close(connection->socket);
    if (connection->addresses) {
        freeaddrinfo( connection->addresses );
    }
    connection->addresses = 0;
    connection->picked_address = 0;
}

Result
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

Result create_thread(ThreadProc proc, void* user_data, Thread* thread) {
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

void
join_thread(Thread* thread) {
#ifdef _WIN32
    WaitForSingleObject(thread->handle, INFINITE);
    CloseHandle(thread->handle);
#else
    pthread_join(thread->thread, 0);
#endif
}

typedef atomic_uint Futex;

inline void futex_init(Futex* futex, uint32_t initial_value) {
    atomic_store_explicit(futex, initial_value, memory_order_relaxed);
}

// Could return false on timeout (not implemented yet)
inline bool futex_wait(Futex* futex, uint32_t expected) {
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
inline bool futex_wake(Futex* futex) {
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

inline void futex_wake_all(Futex* futex) {
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

inline bool _rwlock_is_unlocked(uint32_t state) {
    return (state & RWLOCK_MASK) == 0;
}

inline bool _rwlock_is_write_locked(uint32_t state) {
    return (state & RWLOCK_MASK) == RWLOCK_WRITE_LOCKED;
}

inline bool _rwlock_has_readers_waiting(uint32_t state) {
    return (state & RWLOCK_READERS_WAITING) != 0;
}

inline bool _rwlock_has_writers_waiting(uint32_t state) {
    return (state & RWLOCK_WRITERS_WAITING) != 0;
}

inline bool _rwlock_is_read_lockable(uint32_t state) {
    // This also returns false if the counter could overflow if we tried to read lock it.
    //
    // We don't allow read-locking if there's readers waiting, even if the lock is unlocked
    // and there's no writers waiting. The only situation when this happens is after unlocking,
    // at which point the unlocking thread might be waking up writers, which have priority over readers.
    // The unlocking thread will clear the readers waiting bit and wake up readers, if necessary.
    return (state & RWLOCK_MASK) < RWLOCK_MAX_READERS && !_rwlock_has_readers_waiting(state) && !_rwlock_has_writers_waiting(state);
}

inline bool _rwlock_is_read_lockable_after_wakeup(uint32_t state){
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

inline bool _rwlock_has_reached_max_readers(uint32_t state) {
    return (state & RWLOCK_MASK) == RWLOCK_MAX_READERS;
}


inline void rwlock_init(RWLock* rwlock) {
    futex_init(&rwlock->state, 0);
    futex_init(&rwlock->writer_notify, 0);
}

inline bool rwlock_try_read(RWLock* rwlock) {
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

inline static void rwlock_read_unlock(RWLock* rwlock) {
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

void free_ring_mapped_buffer(void* ptr, size_t size) {
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

// Serialization

// Tracing enabled semaphore

// Sqlite

// Subscribers

// SQLite subscriber

// TCP subscriber

// Tracer

// Default global tracer

#endif

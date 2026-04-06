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

#ifdef __APPLE__
#include <os/os_sync_wait_on_address.h>
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
    sprintf( portbuf, "%" PRIu16, port );
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
    sprintf( portbuf, "%" PRIu16, port );

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
    sprintf( portbuf, "%" PRIu16, port );

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
    return WaitOnAddress(futex, &expected, 4, INFINITE) == TRUE;
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
        if (atomic_load_explicit(futex, memory_order_relaxed) == expected) {
            break;
        }

        int ret = (futex, expected, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE);
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
    WakeByAddressSingle(futex);
    return false
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
    WakeByAddressAll(futex);
#elif defined(__APPLE__)
    os_sync_wake_by_address_all(futex, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
#else
    syscall(SYS_futex, futex, FUTEX_WAKE | FUTEX_PRIVATE_FLAG, INT32_MAX) > 0;
#endif
}

#define RWLOCK_READ_LOCKED        ((uint32_t)1)
#define RWLOCK_MASK               ((uint32_t)((1 << 30) - 1))
#define RWLOCK_WRITE_LOCKED       RWLOCK_MASK
#define RWLOCK_DOWNGRADE          ((uint32_t)(RWLOCK_READ_LOCKED - RWLOCK_WRITE_LOCKED))
#define RWLOCK_MAX_READERS        ((uint32_t)(RWLOCK_MASK - 1))
#define RWLOCK_READERS_WAITING    ((uint32_t)(1 << 30))
#define RWLOCK_WRITERS_WAITING    ((uint32_t)(1 << 31))

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
        if (_rwlock_is_read_lockable(prev)) {
            uint32_t desired = prev + RWLOCK_READ_LOCKED;
            if (atomic_compare_exchange_weak_explicit(&rwlock->state, &prev, desired, memory_order_acquire, memory_order_relaxed)) {
                return true;
            }
        }
    }
    return false;
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

inline uint32_t _rwlock_spin_write(RWLock* rwlock) {
    size_t spin = 100;
    while(true) {
        uint32_t state = atomic_load_explicit(&rwlock->state, memory_order_relaxed);

        // Stop spinning when it's unlocked or when there's waiting writers, to keep things somewhat fair.
        if ((_rwlock_is_unlocked(state) || _rwlock_has_writers_waiting(state)) || spin == 0) {
            return state;
        }
        SpinlockHint();
    }
}

inline uint32_t _rwlock_spin_read(RWLock* rwlock) {
    size_t spin = 100;
    while(true) {
        uint32_t state = atomic_load_explicit(&rwlock->state, memory_order_relaxed);

        // Stop spinning when it's unlocked or when there's waiting writers, to keep things somewhat fair.
        if ((!_rwlock_is_write_locked(state) || _rwlock_has_readers_waiting(state) || _rwlock_has_writers_waiting(state)) || spin == 0) {
            return state;
        }
        SpinlockHint();
    }
}

inline void _rwlock_read_contended(RWLock* rwlock) {
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
            uint32_t desired = state + RWLOCK_READERS_WAITING;
            if (!atomic_compare_exchange_weak_explicit(&rwlock->state, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
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

inline void rwlock_read(RWLock* rwlock) {
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
inline bool _rwlock_wake_writer(RWLock* rwlock) {
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

inline bool rwlock_try_write(RWLock* rwlock) {
    uint32_t prev = atomic_load_explicit(&rwlock->state, memory_order_acquire);
    while (true) {
        if (_rwlock_is_unlocked(prev)) {
            uint32_t desired = prev + RWLOCK_WRITE_LOCKED;
            if (atomic_compare_exchange_weak_explicit(&rwlock->state, &prev, desired, memory_order_acquire, memory_order_relaxed)) {
                return true;
            }
        }
    }
    return false;
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

inline void downgrade(RWLock* rwlock) {
    // Removes all write bits and adds a single read bit.
    uint32_t state = atomic_fetch_add_explicit(&rwlock->state, RWLOCK_DOWNGRADE, memory_order_release);

    // debug_assert(_rwlock_is_write_locked(state), "RwLock must be write locked to call `downgrade`");

    if (_rwlock_has_readers_waiting(state)) {
        // Since we had the exclusive lock, nobody else can unset this bit.
        atomic_fetch_sub_explicit(&rwlock->state, RWLOCK_READERS_WAITING, memory_order_relaxed);
        futex_wake_all(&rwlock->state);
    }
}

// Pre-allocated MPSC ringbuffer with wait and drop semantics
typedef struct MpscRingBuffer
{
    size_t allocated_offset;
    size_t written_offset;
    size_t read_offset;

    // Mutex writer_mutex;
    // ConditionVariable writer_condition_variable;

    // Mutex reader_mutex;
    // ConditionVariable reader_condition_variable;

    uint8_t* buffer;
    size_t size;
} MpscRingBuffer;

// Reader writer lock for subscribers

// Serialization

// Tracing enabled semaphore

// Sqlite

// Subscribers

// SQLite subscriber

// TCP subscriber

// Tracer

// Default global tracer

#endif

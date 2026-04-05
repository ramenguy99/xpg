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
            const auto err = errno;
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
    int result = pthread_create(&thread->thread, 0, proc, data);
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

// MPSC ringbuffer with switchable wait and drop semantics

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

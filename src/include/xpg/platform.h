#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <pthread.h>
#endif

#include "array.h"

namespace xpg {
namespace platform {

struct File {
#ifdef _WIN32
    HANDLE handle;
#else
    int fd;
#endif
    u64 size;
};

struct FileReadWork {
    // Submission info
    File file;
    u64 offset;
    ArrayView<u8> buffer;
    bool do_chunks;

    // Work state
    u64 bytes_read;
};

enum class Result {
    Success,
    GetFileSizeError,
    FileNotFound,
    IOError,
    OutOfBounds,
    OSError,
};

Result OpenFile(const char* path, File* result);
Result ReadAtOffset(File file, ArrayView<u8> buffer, u64 offset);
Result ReadExact(File file, ArrayView<u8> data);
Result ReadEntireFile(const char* path, Array<u8>*data);

struct Timestamp {
    u64 value;
};

Timestamp GetTimestamp();
f64 GetElapsed(Timestamp begin, Timestamp end);

struct SystemTime {
    uint32_t year;
    uint32_t month;
    uint32_t day;
    uint32_t hour;
    uint32_t minute;
    uint32_t second;
    uint32_t milliseconds;
};
Result GetLocalTime(SystemTime* s);

} // namespace platform
} // namespace xpg
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

File OpenFile(const char* path) {
#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    assert(file != INVALID_HANDLE_VALUE);

    LARGE_INTEGER size_large = {};
    BOOL ok = GetFileSizeEx(file, &size_large);
    assert(ok);

    File result = {};
    result.handle = file;
    result.size = size_large.QuadPart;
#else
    int fd = open(path, O_RDONLY);
    assert(fd >= 0);

    struct stat stat_buf = {};
    int stat_result = fstat(fd, &stat_buf);
    assert(stat_result >= 0);

    File result = {};
    result.fd = fd;
    result.size = stat_buf.st_size;
#endif // _WIN32

    return result;
}

bool ReadAtOffset(File file, ArrayView<u8> buffer, u64 offset) {
    if (offset + buffer.length > file.size) {
        return false;
    }

    u64 size = buffer.length;
    u64 total_read = 0;

    while (total_read < size) {
#ifdef _WIN32
        LONG offset_low = (u32)offset;
        LONG offset_high = offset >> 32;

        OVERLAPPED overlapped = {};
        overlapped.Offset = offset_low;
        overlapped.OffsetHigh = offset_high;

        DWORD bytes_to_read = (DWORD)Min(size - total_read, (u64)0xFFFFFFFF);
        DWORD bread = 0;

        BOOL ok = ReadFile(file.handle, buffer.data + total_read, bytes_to_read, &bread, &overlapped);
#else
        size_t bytes_to_read = size - total_read;
        ssize_t bread = pread(file.fd, buffer.data + total_read, bytes_to_read, offset);
        bool ok = bread >= 0;
#endif // _WIN32
        if (!ok) {
            return false;
        }

        total_read += bread;
        offset += bread;
    }

    return true;
}

Array<u8> ReadEntireFile(const char* path) {
#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    assert(file != INVALID_HANDLE_VALUE);

    LARGE_INTEGER size_large = {};
    BOOL ok = GetFileSizeEx(file, &size_large);
    assert(ok);

    u64 size = size_large.QuadPart;
#else
    int fd = open(path, O_RDONLY);
    assert(fd >= 0);

    struct stat stat_buf = {};
    int stat_result = fstat(fd, &stat_buf);
    assert(stat_result >= 0);

    u64 size = stat_buf.st_size;
#endif

    Array<u8> result(size);
    u64 total_read = 0;
    while (total_read < size) {
#ifdef _WIN32
        DWORD bytes_to_read = (DWORD)Min(size - total_read, (u64)0xFFFFFFFF);
        DWORD bread = 0;

        ok = ReadFile(file, result.data + total_read, bytes_to_read, &bread, 0);
#else 
        size_t bytes_to_read = size - total_read;
        ssize_t bread = read(fd, result.data + total_read, bytes_to_read);
        bool ok = bread >= 0;
#endif
        assert(ok);
        total_read += bread;
    }

    return result;
}

struct Timestamp {
    u64 value;
};

Timestamp GetTimestamp() {
    LARGE_INTEGER l = {};
    QueryPerformanceCounter(&l);

    Timestamp result = {};
    result.value = l.QuadPart;
    return result;
}

f64 GetElapsed(Timestamp begin, Timestamp end) {
    static LARGE_INTEGER frequency = {};
    if (!frequency.QuadPart) {
        QueryPerformanceFrequency(&frequency);
    }

    return (f64)(end.value - begin.value) / (f64)frequency.QuadPart;
}

}

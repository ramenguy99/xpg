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
};

Result OpenFile(const char* path, File* result) {
#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if (file == INVALID_HANDLE_VALUE) {
        return Result::FileNotFound;
    }

    LARGE_INTEGER size_large = {};
    BOOL ok = GetFileSizeEx(file, &size_large);
    if (!ok) {
        return Result::GetFileSizeError;
    }

    u64 size = size_large.QuadPart;
#else
    int file = open(path, O_RDONLY);
    if (file < 0) {
        return Result::FileNotFound;
    }

    struct stat stat_buf = {};
    int stat_result = fstat(file, &stat_buf);
    if (stat_result < 0) {
    }

#endif // _WIN32
    result->handle = file;
    result->size = size_large.QuadPart;

    return Result::Success;
}

Result ReadAtOffset(File file, ArrayView<u8> buffer, u64 offset) {
    if (offset + buffer.length > file.size) {
        return Result::OutOfBounds;
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
            return Result::IOError;
        }

        total_read += bread;
        offset += bread;
    }

    return Result::Success;
}

Result ReadExact(File file, ArrayView<u8> data) {
    u64 total_read = 0;
    while (total_read < data.length) {
#ifdef _WIN32
        DWORD bytes_to_read = (DWORD)Min(data.length - total_read, (u64)0xFFFFFFFF);
        DWORD bread = 0;

        BOOL ok = ReadFile(file.handle, data.data + total_read, bytes_to_read, &bread, 0);
#else 
        size_t bytes_to_read = size - total_read;
        ssize_t bread = read(file.handle, result.data + total_read, bytes_to_read);
        bool ok = bread >= 0;
#endif
        if (!ok) {
            return Result::IOError;
        }
        total_read += bread;
    }

    return Result::Success;
}

Result ReadEntireFile(const char* path, Array<u8>* data) {
    File f = {};
    Result res = OpenFile(path, &f);
    if (res != Result::Success) {
        return res;
    }

    data->resize(f.size);
    return ReadExact(f, *data);
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

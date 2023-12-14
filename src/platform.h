struct File {
    HANDLE handle;
    u64 size;
};

struct FileReadWork {
    File file;
    usize offset;
    ArrayView<u8> buffer;
};

File OpenFile(const char* path) {
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    assert(file != INVALID_HANDLE_VALUE);

    LARGE_INTEGER size_large = {};
    BOOL ok = GetFileSizeEx(file, &size_large);
    assert(ok);

    File result = {};
    result.handle = file;
    result.size = size_large.QuadPart;

    return result;
}

bool ReadAtOffset(File file, ArrayView<u8> buffer, u64 offset) {
    if (offset + buffer.length > file.size) {
        return false;
    }

    u64 size = buffer.length;
    u64 total_read = 0;

    while (total_read < size) {
        LONG offset_low = (u32)offset;
        LONG offset_high = offset >> 32;

        OVERLAPPED overlapped = {};
        overlapped.Offset = offset_low;
        overlapped.OffsetHigh = offset_high;

        DWORD bytes_to_read = (DWORD)Min(size - total_read, (u64)0xFFFFFFFF);
        DWORD bread = 0;

        BOOL ok = ReadFile(file.handle, buffer.data + total_read, bytes_to_read, &bread, &overlapped);
        if (!ok) {
            return false;
        }

        total_read += bread;
    }

    return true;
}

Array<u8> ReadEntireFile(const char* path) {
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    assert(file != INVALID_HANDLE_VALUE);

    LARGE_INTEGER size_large = {};
    BOOL ok = GetFileSizeEx(file, &size_large);
    assert(ok);

    u64 size = size_large.QuadPart;
    Array<u8> result(size);

    u64 total_read = 0;
    while (total_read < size) {
        DWORD bytes_to_read = (DWORD)Min(size - total_read, (u64)0xFFFFFFFF);
        DWORD bread = 0;

        ok = ReadFile(file, result.data + total_read, bytes_to_read, &bread, 0);
        assert(ok);

        total_read += bread;
    }

    return result;
}


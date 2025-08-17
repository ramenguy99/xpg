import io


def read_exact_into(file: io.FileIO, view: memoryview):
    bread = 0
    while bread < len(view):
        n = file.readinto(view[bread:])
        if n == 0:
            raise EOFError()
        else:
            bread += n


def read_exact(file: io.FileIO, size: int):
    out = bytearray(size)
    view = memoryview(out)
    read_exact_into(file, view)
    return out


def read_exact_at_offset_into(file: io.FileIO, offset: int, view: memoryview):
    file.seek(offset, io.SEEK_SET)
    return read_exact_into(file, view)


def read_exact_at_offset(file: io.FileIO, offset: int, size: int):
    file.seek(offset, io.SEEK_SET)
    return read_exact(file, size)

import io


def read_exact_into(file: io.FileIO, view: memoryview) -> None:
    bread = 0
    while bread < len(view):
        n = file.readinto(view[bread:])
        if n == 0:
            raise EOFError
        bread += n


def read_exact(file: io.FileIO, size: int) -> bytearray:
    out = bytearray(size)
    view = memoryview(out)
    read_exact_into(file, view)
    return out


def read_exact_at_offset_into(file: io.FileIO, offset: int, view: memoryview) -> None:
    file.seek(offset, io.SEEK_SET)
    read_exact_into(file, view)


def read_exact_at_offset(file: io.FileIO, offset: int, size: int) -> bytearray:
    file.seek(offset, io.SEEK_SET)
    return read_exact(file, size)

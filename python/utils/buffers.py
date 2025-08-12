from pyxpg import *

class UploadableBuffer(Buffer):
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, name: str = None):
        super().__init__(ctx, size, usage_flags | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE_MAPPED_WITH_FALLBACK, name)
        if not self.is_mapped:
            if name is not None:
                name = f"{name} - staging"
            self._staging = Buffer(ctx, size, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST_WRITE_COMBINING, name)

    def upload(self, cmd: CommandBuffer, usage: MemoryUsage, data: memoryview, offset: int = 0):
        if self.is_mapped:
            self.data[offset:offset+len(data)] = data
        else:
            self._staging.data[offset:offset+len(data)] = data
            cmd.copy_buffer_range(self._staging, self, len(data), src_offset=offset, dst_offset=offset)
            if usage != MemoryUsage.NONE:
                cmd.memory_barrier(MemoryUsage.TRANSFER_DST, usage)

    def upload_sync(self, data: memoryview, offset: int = 0):
        with self.ctx.sync_commands() as cmd:
            self.upload(cmd, MemoryUsage.NONE, data, offset)


from pyxpg import Context, Buffer, BufferUsageFlags, AllocType, CommandBuffer, MemoryUsage
from typing import Optional

class UploadableBuffer(Buffer):
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, name: Optional[str] = None):
        super().__init__(ctx, size, usage_flags | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE_MAPPED_WITH_FALLBACK, name)
        if not self.is_mapped:
            if name is not None:
                name = f"{name} - staging"
            self._staging = Buffer(ctx, size, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST_WRITE_COMBINING, name)
    
    @classmethod
    def from_data(cls, ctx: Context, data: memoryview, usage_flags: BufferUsageFlags, name: Optional[str] = None):
        buf = cls(ctx, len(data), usage_flags, name=name)
        buf.upload_sync(data)
        return buf

    def upload(self, cmd: CommandBuffer, usage: MemoryUsage, data: memoryview, offset: int = 0):
        if self.is_mapped:
            # print(self.data.c_contiguous, data.c_contiguous )
            # print(self.data.shape       , data.shape        )
            # print(self.data.readonly    , data.readonly     )
            # print(self.data.format    , data.format     )
            self.data[offset:offset+len(data)] = data
        else:
            self._staging.data[offset:offset+len(data)] = data
            cmd.copy_buffer_range(self._staging, self, len(data), src_offset=offset, dst_offset=offset)
            if usage != MemoryUsage.NONE:
                cmd.memory_barrier(MemoryUsage.TRANSFER_WRITE, usage)
    
    def upload_sync(self, data: memoryview, offset: int = 0):
        with self.ctx.sync_commands() as cmd:
            self.upload(cmd, MemoryUsage.NONE, data, offset)
